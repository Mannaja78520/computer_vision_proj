import os

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import pandas as pd
import numpy as np
import torch
import keras
import gc
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers, applications, ops

# ตรวจสอบ Backend และระบบ
print(f"\n--- System Check ---")
print(f"Using Backend: {keras.backend.backend()}")
keras.mixed_precision.set_global_policy("mixed_float16") 

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
print(f"--------------------\n")

# ==========================================
# 2. GLOBAL CONFIGURATION
# ==========================================
RANDOM_SEED = 42
VAL_SIZE_INTERNAL = 0.15 
IMG_SIZE = (300, 300)
BATCH_SIZE = 8          
IMAGE_FOLDER = 'all_train_pic/'
MAX_WORKERS = 24

# --- Training Phases ---
PHASE_1_EPOCHS = 5
PHASE_1_LR = 1e-3
PHASE_2_EPOCHS = 100
PHASE_2_LR_START = 3e-5 
EARLY_STOP_PATIENCE = 15 

# --- Model Architecture ---
DENSE_UNIT_1 = 512
DENSE_UNIT_2 = 128
DROPOUT_1 = 0.4         
DROPOUT_2 = 0.3 
MODEL_NAME = 'food_expert_final.keras'

# ==========================================
# 3. DATA PIPELINE (Pro Augmentation)
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
                # 1. Flip ซ้ายขวา
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=1)
                
                # 2. Rotation, Zoom, Shear เบาๆ
                if np.random.rand() > 0.4:
                    img = keras.preprocessing.image.apply_affine_transform(
                        img, theta=np.random.uniform(-10, 10), 
                        zx=np.random.uniform(0.9, 1.1), zy=np.random.uniform(0.9, 1.1),
                        shear=np.random.uniform(-5, 5), fill_mode='nearest'
                    )
                
                # 3. Brightness Adjustment
                if np.random.rand() > 0.4:
                    img = img * np.random.uniform(0.9, 1.1)
                
                img = np.clip(img, 0, 255)

            return applications.efficientnet.preprocess_input(img)
        except:
            return np.zeros((*IMG_SIZE, 3))
    
    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs1 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 1'])), dtype="float32")
        imgs2 = np.array(list(self.executor.map(lambda x: self.load_img(x, self.augment), batch['Image 2'])), dtype="float32")
        labels = batch['target'].values.astype("float32")
        weights = batch['weight'].values.astype("float32")
        return [imgs1, imgs2], labels, weights

# --- การเตรียมข้อมูลพร้อมแสดงสถิติ ---
df_all_train = pd.read_csv('train_split.csv')
total_raw = len(df_all_train)

df_t_raw, df_v = train_test_split(df_all_train, test_size=VAL_SIZE_INTERNAL, random_state=RANDOM_SEED)

def augment_siamese_df(df):
    df_swapped = df.copy()
    df_swapped['Image 1'], df_swapped['Image 2'] = df['Image 2'], df['Image 1']
    df_swapped['target'] = 1.0 - df['target']
    return pd.concat([df, df_swapped], ignore_index=True)

df_train_augmented = augment_siamese_df(df_t_raw).sample(frac=1, random_state=RANDOM_SEED)
df_val_augmented = augment_siamese_df(df_v)

train_ds = SiameseDataset(df_train_augmented, augment=True)
val_ds = SiameseDataset(df_val_augmented, augment=False)

print(f"\n" + "="*45)
print(f"📊 DATASET SUMMARY (Balanced Version)")
print(f"="*45)
print(f"Total Raw Pairs from CSV: {total_raw}")
print(f"Training Set ({(1-VAL_SIZE_INTERNAL) * 100}%):   {len(df_t_raw)} -> After Swap Clone: {len(df_train_augmented)}")
print(f"Validation Set ({VAL_SIZE_INTERNAL * 100}%): {len(df_v)} -> After Swap Clone: {len(df_val_augmented)}")
print(f"-" * 45)
print(f"Batch Size:  {BATCH_SIZE}")
print(f"Steps per Epoch: {int(np.ceil(len(df_train_augmented) / BATCH_SIZE))}")
print(f"*" * 45)
print(f"AI will train on {len(df_train_augmented)} pairs and validate on {len(df_val_augmented)} pairs.")
print(f"Advanced Image Augmentation: ENABLED (Symmetry/Rotate/Zoom)")
print(f"="*45 + "\n")

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
def build_model():
    base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1 = layers.Input(shape=(*IMG_SIZE, 3))
    input_2 = layers.Input(shape=(*IMG_SIZE, 3))
    
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    bn_entry = layers.BatchNormalization()
    feat_1 = bn_entry(feat_1)
    feat_2 = bn_entry(feat_2)
    
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, mul])
    
    x = layers.Dense(DENSE_UNIT_1, activation='swish')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_1)(x)
    
    x = layers.Dense(DENSE_UNIT_2, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_2)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# ==========================================
# 5. TRAINING EXECUTION
# ==========================================
class MemoryCleanCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        torch.cuda.empty_cache()
        gc.collect()

keras.backend.clear_session()
model = build_model()

model.compile(optimizer=optimizers.Adam(PHASE_1_LR), loss='binary_crossentropy', metrics=['mae'])
print(f"[Phase 1] Starting...")
model.fit(train_ds, validation_data=val_ds, epochs=PHASE_1_EPOCHS, callbacks=[MemoryCleanCallback()])

print(f"\n[Phase 2] Fine-tuning all layers (Balanced Strategy)...")
for layer in model.layers: layer.trainable = True
model.compile(optimizer=optimizers.Adam(learning_rate=PHASE_2_LR_START), loss='binary_crossentropy', metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=PHASE_2_EPOCHS, callbacks=[MemoryCleanCallback(), early_stop])

model.save(MODEL_NAME)
print(f"\nTraining Complete! Model saved as {MODEL_NAME}")