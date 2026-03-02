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

# --- 2. SELECT BACKBONE TYPE ---
# เลือกเปลี่ยนตรงนี้ก่อนรันเทรนแต่ละชุด: 'v2s', 'convnext', 'v2m'
BACKBONE_TYPE = 'v2s' 

# --- 3. CONFIGURATION ---
RANDOM_SEED = 42
K_FOLDS = 5         
IMG_SIZE = (300, 300)
BATCH_SIZE = 6      # ลดลงนิดหน่อยเพื่อให้ V2M (ตัวใหญ่) เทรนบน Laptop ได้
IMAGE_FOLDER = 'all_train_pic/'
MAX_WORKERS = 16

PHASE_1_EPOCHS = 5
PHASE_2_EPOCHS = 100
EARLY_STOP_PATIENCE = 12

# ==========================================
# 4. MODEL BUILDER (Dynamic Backbone)
# ==========================================
def build_model(model_type):
    if model_type == 'v2s':
        base = applications.EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
    elif model_type == 'convnext':
        base = applications.ConvNeXtTiny(weights='imagenet', include_top=False, pooling='avg')
    elif model_type == 'v2m':
        base = applications.EfficientNetV2M(weights='imagenet', include_top=False, pooling='avg')
    
    base.trainable = False 
    
    input_1 = layers.Input(shape=(*IMG_SIZE, 3))
    input_2 = layers.Input(shape=(*IMG_SIZE, 3))
    
    feat_1, feat_2 = base(input_1), base(input_2)
    
    shared_bn = layers.BatchNormalization()
    feat_1, feat_2 = shared_bn(feat_1), shared_bn(feat_2)
    
    # Interaction Layers
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    sq_diff = layers.Lambda(lambda x: ops.square(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, sq_diff, mul])
    
    l2_reg = regularizers.l2(2e-4)
    x = layers.Dense(1024, activation='swish', kernel_regularizer=l2_reg)(merged)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.45)(x)
    x = layers.Dense(512, activation='swish', kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.35)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# ==========================================
# 5. DATA PIPELINE (Dynamic Preprocessing)
# ==========================================
class SiameseDataset(keras.utils.PyDataset):
    def __init__(self, df, model_type, batch_size=BATCH_SIZE, augment=False, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.model_type = model_type
        self.batch_size = batch_size
        self.augment = augment
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def load_img(self, name, augment=False):
        try:
            path = os.path.join(IMAGE_FOLDER, str(name))
            img = keras.utils.load_img(path, target_size=IMG_SIZE)
            img = keras.utils.img_to_array(img)
            if augment:
                if np.random.rand() > 0.5: img = np.flip(img, axis=1)
                img *= np.random.uniform(0.9, 1.1) 
                img = np.clip(img, 0, 255)
            
            # เลือก Preprocess ให้ตรงสายพันธุ์
            if self.model_type == 'v2s':
                return applications.efficientnet_v2.preprocess_input(img)
            elif self.model_type == 'convnext':
                return applications.convnext.preprocess_input(img)
            elif self.model_type == 'v2m':
                return applications.efficientnet_v2.preprocess_input(img)
            return img / 255.0
        except: return np.zeros((*IMG_SIZE, 3))
    
    def __len__(self): return int(np.ceil(len(self.df) / self.batch_size))
    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs1 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 1'])), dtype="float32")
        imgs2 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 2'])), dtype="float32")
        labels = batch['target'].values.astype("float32")
        if self.augment and np.random.rand() > 0.5: return [imgs2, imgs1], 1.0 - labels
        return [imgs1, imgs2], labels

# ==========================================
# 6. TRAINING PROCESS
# ==========================================
df1 = pd.read_csv('train_split.csv')
df1['target'] = np.where(df1['Winner'] == 1, 0.12, 0.88)
df2 = pd.read_csv('data_from_questionaire.csv')
df2['target'] = df2['Num Vote 2'] / df2['Num Voter']
df_all = pd.concat([df1, df2], ignore_index=True)

kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print(f"🚀 Training Hybrid Strategy: {BACKBONE_TYPE}")

for fold, (train_idx, val_idx) in enumerate(kf.split(df_all)):
    # กำหนดช่วง Fold ตามแผน (v2s=Fold 1-2, convnext=Fold 3-4, v2m=Fold 5)
    current_fold = fold + 1
    target_type = BACKBONE_TYPE
    
    # --- เพิ่มบรรทัดนี้เพื่อข้าม Fold ที่เทรนเสร็จแล้ว ---
    if current_fold < 5:
        print(f"⏩ Skipping Fold {current_fold} (Already trained)")
        continue
    # ----------------------------------------------
    
    # ถ้าอยากรันทีเดียวให้เลือก Backbone ตาม Fold (Manual Override)
    if current_fold in [1, 2]: target_type = 'v2s'
    elif current_fold in [3, 4]: target_type = 'convnext'
    else: target_type = 'v2m'

    print(f"\n🚩 FOLD {current_fold} | Backbone: {target_type}")
    keras.backend.clear_session(); gc.collect(); torch.cuda.empty_cache()
    
    df_train, df_val = df_all.iloc[train_idx], df_all.iloc[val_idx]
    
    def augment_df(df):
        sw = df.copy(); sw['Image 1'], sw['Image 2'] = df['Image 2'], df['Image 1']
        sw['target'] = 1.0 - df['target']
        return pd.concat([df, sw], ignore_index=True)

    train_ds = SiameseDataset(augment_df(df_train).sample(frac=1), target_type, augment=True)
    val_ds = SiameseDataset(df_val, target_type, augment=False)
    
    model = build_model(target_type)
    
    # Phase 1
    model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['mae'])
    model.fit(train_ds, validation_data=val_ds, epochs=PHASE_1_EPOCHS)
    
    # Phase 2: Fine-tuning
    for layer in model.layers: layer.trainable = True
    lr_sch = optimizers.schedules.CosineDecay(4e-5, len(train_ds)*PHASE_2_EPOCHS, alpha=0.1)
    model.compile(optimizer=optimizers.Adam(lr_sch), loss='binary_crossentropy', metrics=['mae'])
    
    stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
    model.fit(train_ds, validation_data=val_ds, epochs=PHASE_2_EPOCHS, callbacks=[stop])
    
    model.save(f'fold_{current_fold}_{target_type}_final.keras')

print("\n✅ All 5 Diverse Folds Training Completed!")