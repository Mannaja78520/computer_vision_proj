import os
import sys

# --- 1. SET ENVIRONMENT FIRST (ต้องทำก่อน Import อื่นๆ) ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import pandas as pd
import numpy as np
import torch
import keras
import gc
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers, applications, ops

# --- 2. GPU & PRECISION SETUP ---
keras.mixed_precision.set_global_policy("mixed_float16") 
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print(f"Using Backend: {keras.backend.backend()}")
    print(f"GPU Optimized for: {torch.cuda.get_device_name(0)}")

# --- 3. GLOBAL CONFIGURATION ---
RANDOM_SEED = 42
IMG_SIZE = (300, 300)
BATCH_SIZE = 8  # Blackwell (RTX 5070) รับ Batch ขนาดนี้ได้สบายและเสถียรกว่า
IMAGE_FOLDER = 'all_train_pic/'
MAX_WORKERS = 16

PHASE_1_EPOCHS = 5
PHASE_2_EPOCHS = 80
EARLY_STOP_PATIENCE = 10

# ==========================================
# 4. DATA PIPELINE (Symmetry & Parallel Loading)
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
                if np.random.rand() > 0.5: img = np.flip(img, axis=1) # Horizontal Flip
                img *= np.random.uniform(0.9, 1.1) # Brightness
                img = np.clip(img, 0, 255)

            return applications.efficientnet_v2.preprocess_input(img)
        except:
            return np.zeros((*IMG_SIZE, 3))
    
    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs1 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 1'])), dtype="float32")
        imgs2 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 2'])), dtype="float32")
        labels = batch['target'].values.astype("float32")
        
        # --- KEY: In-Batch Symmetry (บังคับให้โมเดลไม่จำตำแหน่ง) ---
        if self.augment and np.random.rand() > 0.5:
            return [imgs2, imgs1], 1.0 - labels
            
        return [imgs1, imgs2], labels

# --- 5. DATA PREPARATION ---
def augment_df(df):
    df_swapped = df.copy()
    df_swapped['Image 1'], df_swapped['Image 2'] = df['Image 2'], df['Image 1']
    df_swapped['target'] = 1.0 - df['target']
    return pd.concat([df, df_swapped], ignore_index=True)

df1 = pd.read_csv('train_split.csv')
df1['target'] = np.where(df1['Winner'] == 1, 0.05, 0.95)
df2 = pd.read_csv('data_from_questionaire.csv')
df2['target'] = df2['Num Vote 2'] / df2['Num Voter']

df_all = pd.concat([df1, df2], ignore_index=True)
df_train_raw, df_val = train_test_split(df_all, test_size=0.1, random_state=RANDOM_SEED)

train_ds = SiameseDataset(augment_df(df_train_raw).sample(frac=1), augment=True)
val_ds = SiameseDataset(df_val, augment=False)

# ==========================================
# 6. MODEL ARCHITECTURE (EfficientNetV2-S + Shared BN)
# ==========================================
def build_model():
    base_model = applications.EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1 = layers.Input(shape=(*IMG_SIZE, 3))
    input_2 = layers.Input(shape=(*IMG_SIZE, 3))
    
    # Shared Backbone
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    # Shared BatchNormalization (กุญแจสำคัญของความสมมาตร)
    shared_bn = layers.BatchNormalization()
    feat_1 = shared_bn(feat_1)
    feat_2 = shared_bn(feat_2)
    
    # Interaction Layers
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    sq_diff = layers.Lambda(lambda x: ops.square(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, sq_diff, mul])
    
    # Dense Head
    x = layers.Dense(512, activation='swish')(merged)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# ==========================================
# 7. TRAINING EXECUTION
# ==========================================
keras.backend.clear_session()
model = build_model()

# Phase 1: Foundation
model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['mae'])
print("\n🚩 Phase 1: Training Head...")
model.fit(train_ds, validation_data=val_ds, epochs=PHASE_1_EPOCHS)

# Phase 2: Fine-tuning
print("\n🚩 Phase 2: Fine-tuning Full Model...")
for layer in model.layers: layer.trainable = True

lr_schedule = optimizers.schedules.ExponentialDecay(2e-5, 1000, 0.9, staircase=True)
model.compile(optimizer=optimizers.Adam(lr_schedule), loss='binary_crossentropy', metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=PHASE_2_EPOCHS, callbacks=[early_stop])

model.save('food_expert_v2s_symmetric.keras')
print("\n✅ Complete! Model saved as 'food_expert_v2s_symmetric.keras'")