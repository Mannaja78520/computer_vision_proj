import os

# --- 1. บังคับตั้งค่าสภาพแวดล้อม "ก่อน" Import Library อื่นๆ ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# --- 2. จากนั้นจึง Import Library หลักตามลำดับนี้ ---
import pandas as pd
import numpy as np
import torch
import keras
import gc
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers, applications, ops
# ---------------------------------------------------------

# ตรวจสอบทันทีว่า Backend ถูกต้องไหม
print(f"Using Backend: {keras.backend.backend()}") # ต้องขึ้นว่า torch

# ตั้งค่า Policy สำหรับ GPU 
keras.mixed_precision.set_global_policy("mixed_float16") 

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    

# ==========================================
# 0. GLOBAL CONFIGURATION (ปรับจูนทั้งหมดที่นี่)
# ==========================================
# --- Data Settings ---
RANDOM_SEED = 42
VAL_SIZE = 0.1
IMG_SIZE = (300, 300)
BATCH_SIZE = 6
IMAGE_FOLDER = 'all_train_pic/'
MAX_WORKERS = 24

# --- Phase 1: Foundation (Freeze Base) ---
PHASE_1_EPOCHS = 5
PHASE_1_LR = 1e-3

# --- Phase 2: Fine-tuning (Unfreeze All) ---
PHASE_2_EPOCHS = 60
PHASE_2_LR_START = 5e-5
LR_DECAY_RATE = 0.9
LR_DECAY_STEPS = 1000
EARLY_STOP_PATIENCE = 10 #จำนวน Epoch ที่จะรอหลังจากไม่มีการปรับปรุงใน Validation MAE ก่อนที่จะหยุดการฝึก

# --- Model Architecture ---
DENSE_UNIT_1 = 512
DENSE_UNIT_2 = 128
DROPOUT_1 = 0.4
DROPOUT_2 = 0.3
MODEL_NAME = 'food_expert_final.keras'

# ==========================================
# 1. ENVIRONMENT SETUP
# ==========================================
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

keras.mixed_precision.set_global_policy("mixed_float16") 

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print(f"Using Backend: {keras.backend.backend()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ==========================================
# 2. DATA PIPELINE
# ==========================================
class SiameseDataset(keras.utils.PyDataset):
    def __init__(self, df, batch_size=BATCH_SIZE, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size)) 

    def load_img(self, name, augment=False):
        try:
            path = os.path.join(IMAGE_FOLDER, str(name))
            img = keras.utils.load_img(path, target_size=IMG_SIZE)
            img = keras.utils.img_to_array(img)
            
            if augment:
                if np.random.rand() > 0.5:
                    img = np.flip(img, axis=1)
                img = img * np.random.uniform(0.8, 1.2)
                img = np.clip(img, 0, 255)

            return applications.efficientnet.preprocess_input(img)
        except:
            return np.zeros((*IMG_SIZE, 3))
    
    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs1 = np.array(list(self.executor.map(self.load_img, batch['Image 1'], [True]*len(batch))), dtype="float32")
        imgs2 = np.array(list(self.executor.map(self.load_img, batch['Image 2'], [True]*len(batch))), dtype="float32")
        labels = batch['target'].values.astype("float32")
        return [imgs1, imgs2], labels

# --- การเตรียม DataFrames ---
df1_raw = pd.read_csv('train_split.csv') 
df1_raw['target'] = np.where(df1_raw['Winner'] == 1, 0.05, 0.95)

df2_raw = pd.read_csv('data_from_questionaire.csv')
df2_raw['target'] = df2_raw['Num Vote 2'] / df2_raw['Num Voter']

df_all_raw = pd.concat([df1_raw, df2_raw], ignore_index=True)
df_train_raw, df_val = train_test_split(df_all_raw, test_size=VAL_SIZE, random_state=RANDOM_SEED)

def augment_siamese(df):
    df_swapped = df.copy()
    df_swapped['Image 1'], df_swapped['Image 2'] = df['Image 2'], df['Image 1']
    df_swapped['target'] = 1.0 - df['target']
    return pd.concat([df, df_swapped], ignore_index=True)

# Dataset Instances
train_ds_p1 = SiameseDataset(augment_siamese(df1_raw).sample(frac=1, random_state=RANDOM_SEED))
train_ds_p2 = SiameseDataset(augment_siamese(df_train_raw).sample(frac=1, random_state=RANDOM_SEED))
val_ds_p2 = SiameseDataset(df_val)

# ==========================================
# 3. MODEL ARCHITECTURE
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
    
    # Dense Block 1
    x = layers.Dense(DENSE_UNIT_1, activation='swish')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_1)(x)
    
    # Dense Block 2
    x = layers.Dense(DENSE_UNIT_2, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_2)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# ==========================================
# 4. TRAINING EXECUTION
# ==========================================
class MemoryCleanCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        torch.cuda.empty_cache()
        gc.collect()

keras.backend.clear_session()
model = build_model()

# --- PHASE 1 ---
model.compile(optimizer=optimizers.Adam(PHASE_1_LR), loss='binary_crossentropy', metrics=['mae'])
print(f"\n[Phase 1] Training Foundation for {PHASE_1_EPOCHS} epochs...")
model.fit(train_ds_p1, validation_data=val_ds_p2, epochs=PHASE_1_EPOCHS, callbacks=[MemoryCleanCallback()])

# --- PHASE 2 ---
print(f"\n[Phase 2] Fine-tuning all layers for {PHASE_2_EPOCHS} epochs...")
for layer in model.layers:
    layer.trainable = True

lr_schedule = optimizers.schedules.ExponentialDecay(
    PHASE_2_LR_START, decay_steps=LR_DECAY_STEPS, decay_rate=LR_DECAY_RATE, staircase=True
)
model.compile(optimizer=optimizers.Adam(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['mae'])

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_mae', 
    patience=EARLY_STOP_PATIENCE, 
    restore_best_weights=True
)

model.fit(
    train_ds_p2, 
    validation_data=val_ds_p2, 
    epochs=PHASE_2_EPOCHS, 
    callbacks=[MemoryCleanCallback(), early_stop]
)

model.save(MODEL_NAME)
print(f"\nTraining Complete! Model saved as {MODEL_NAME}")