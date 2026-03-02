import os
import sys

# --- 1. SET ENVIRONMENT ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import pandas as pd
import numpy as np
import torch
import keras
import gc
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import KFold
from keras import layers, models, optimizers, applications, ops, regularizers

# --- 2. GPU & PRECISION ---
keras.mixed_precision.set_global_policy("mixed_float16") 
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()

# --- 3. CONFIGURATION ---
RANDOM_SEED = 42
K_FOLDS = 5         # กลับมาทำ 5-Fold เพื่อความแม่นยำสูงสุด
IMG_SIZE = (300, 300)
BATCH_SIZE = 8 
IMAGE_FOLDER = 'all_train_pic/'
MAX_WORKERS = 16

PHASE_1_EPOCHS = 5
PHASE_2_EPOCHS = 80
EARLY_STOP_PATIENCE = 10

# ==========================================
# 4. DATA PIPELINE
# ==========================================
class SiameseDataset(keras.utils.PyDataset):
    def __init__(self, df, batch_size=BATCH_SIZE, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.batch_size = batch_size
        self.augment = augment
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size)) 

    def load_img(self, name, augment=False):
        try:
            path = os.path.join(IMAGE_FOLDER, str(name))
            img = keras.utils.load_img(path, target_size=IMG_SIZE)
            img = keras.utils.img_to_array(img)
            if augment:
                if np.random.rand() > 0.5: img = np.flip(img, axis=1)
                img *= np.random.uniform(0.9, 1.1) 
                img = np.clip(img, 0, 255)
            return applications.efficientnet_v2.preprocess_input(img)
        except: return np.zeros((*IMG_SIZE, 3))
    
    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs1 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 1'])), dtype="float32")
        imgs2 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 2'])), dtype="float32")
        labels = batch['target'].values.astype("float32")
        if self.augment and np.random.rand() > 0.5:
            return [imgs2, imgs1], 1.0 - labels
        return [imgs1, imgs2], labels

# ==========================================
# 5. MODEL ARCHITECTURE (L2 + 1024/512)
# ==========================================
def build_model():
    base_model = applications.EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1 = layers.Input(shape=(*IMG_SIZE, 3))
    input_2 = layers.Input(shape=(*IMG_SIZE, 3))
    
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    shared_bn = layers.BatchNormalization()
    feat_1 = shared_bn(feat_1)
    feat_2 = shared_bn(feat_2)
    
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    sq_diff = layers.Lambda(lambda x: ops.square(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, sq_diff, mul])
    
    # เพิ่ม L2 Regularizer เพื่อคุม Overfit
    l2_reg = regularizers.l2(1e-4)
    
    x = layers.Dense(1024, activation='swish', kernel_regularizer=l2_reg)(merged)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='swish', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# ==========================================
# 6. DATA PREPARATION (Label Smoothing 0.1)
# ==========================================
def augment_df(df):
    df_swapped = df.copy()
    df_swapped['Image 1'], df_swapped['Image 2'] = df['Image 2'], df['Image 1']
    df_swapped['target'] = 1.0 - df['target']
    return pd.concat([df, df_swapped], ignore_index=True)

df1 = pd.read_csv('train_split.csv')
# ปรับเป็น 0.1 / 0.9 เพื่อให้โมเดล Generalize เก่งขึ้น
df1['target'] = np.where(df1['Winner'] == 1, 0.1, 0.9)
df2 = pd.read_csv('data_from_questionaire.csv')
df2['target'] = df2['Num Vote 2'] / df2['Num Voter']
df_all = pd.concat([df1, df2], ignore_index=True)

# ==========================================
# 7. 5-FOLD EXECUTION
# ==========================================
kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)
fold_histories = []

print(f"🚀 Starting {K_FOLDS}-Fold CV Hybrid Training...")

for fold, (train_idx, val_idx) in enumerate(kf.split(df_all)):
    print(f"\n🚩 FOLD {fold + 1}/{K_FOLDS}")
    keras.backend.clear_session(); gc.collect(); torch.cuda.empty_cache()
    
    df_train = df_all.iloc[train_idx]
    df_val = df_all.iloc[val_idx]
    
    train_ds = SiameseDataset(augment_df(df_train).sample(frac=1), augment=True)
    val_ds = SiameseDataset(df_val, augment=False)
    
    model = build_model()
    
    # Phase 1
    model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['mae'])
    model.fit(train_ds, validation_data=val_ds, epochs=PHASE_1_EPOCHS, verbose=1)
    
    # Phase 2
    for layer in model.layers: layer.trainable = True
    lr_schedule = optimizers.schedules.ExponentialDecay(3e-5, 1000, 0.9, staircase=True)
    model.compile(optimizer=optimizers.Adam(lr_schedule), loss='binary_crossentropy', metrics=['mae'])
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=PHASE_2_EPOCHS, callbacks=[early_stop], verbose=1)
    
    model.save(f'fold_{fold+1}_v2s_final.keras')
    fold_histories.append(min(history.history['val_mae']))
    print(f"✅ Fold {fold+1} Done. Val MAE: {fold_histories[-1]:.4f}")

print(f"\n🏆 Average Val MAE: {np.mean(fold_histories):.4f}")
print("All folds completed. Use Ensemble in Predict script tomorrow!")