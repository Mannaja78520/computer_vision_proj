import os
import sys

# --- 1. ตั้งค่า Backend แบบเด็ดขาด (ต้องไว้บนสุดก่อน import อื่นๆ) ---
# --- 1. ตั้งค่า Backend และ Memory ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # ปิดการแจ้งเตือน TensorFlow ทั้งหมด
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # ปิดตัวเร่งคำนวณที่อาจขัดแย้งกับ CUDA ใหม่

import pandas as pd
import numpy as np
import torch
import keras
import gc
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from keras import layers, models, optimizers, applications, ops

# ตรวจสอบว่า Keras ใช้ Backend อะไร (ต้องขึ้นว่า torch)
print(f"Using Backend: {keras.backend.backend()}")

# บังคับให้ PyTorch จัดการ Memory แบบ Blackwell-compatible
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

# ปิด Mixed Precision ชั่วคราว (ถ้ายัง Error อยู่) 
# เพราะ Blackwell (RTX 50) มีการจัดการ Casting ที่ต่างจากรุ่นก่อน
keras.mixed_precision.set_global_policy("mixed_float16") 

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    torch.cuda.empty_cache()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
# ==========================================
# 0. CONFIGURATION (ปรับแต่งง่ายๆ ที่นี่)
# ==========================================
RANDOM_SEED = 42
VAL_SIZE = 0.1  # แบ่ง Validation 10% (Ratio 9:1)
IMG_SIZE = (300, 300)
BATCH_SIZE = 6
IMAGE_FOLDER = 'all_train_pic/'
MAX_WORKERS = 24
PHASE_1_EPOCHS = 7
PHASE_2_EPOCHS = 45

# ==========================================
# 2. DATA PIPELINE
# ==========================================
class SiameseDataset(keras.utils.PyDataset):
    def __init__(self, df, batch_size=6, **kwargs):
        super().__init__(**kwargs)
        self.df = df
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size)) 

    def load_img(self, name):
        try:
            img_path = os.path.join(IMAGE_FOLDER, str(name))
            img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img = keras.utils.img_to_array(img)
            return applications.efficientnet.preprocess_input(img)
        except:
            return np.zeros((*IMG_SIZE, 3))

    def __getitem__(self, idx):
        batch = self.df[idx * self.batch_size : (idx + 1) * self.batch_size]
        imgs1 = np.array(list(self.executor.map(self.load_img, batch['Image 1'])), dtype="float32")
        imgs2 = np.array(list(self.executor.map(self.load_img, batch['Image 2'])), dtype="float32")
        labels = batch['target'].values.astype("float32")
        return [imgs1, imgs2], labels

# --- เตรียมข้อมูล ---
# 1. โหลดข้อมูลดิบ "ก่อนสลับ"
# ใช้ไฟล์ต้นฉบับเพื่อให้การแบ่ง 9:1 ใสสะอาดที่สุด
df1_raw = pd.read_csv('train_split.csv') 
df1_raw['target'] = np.where(df1_raw['Winner'] == 1, 0.05, 0.95)

df2_raw = pd.read_csv('data_from_questionaire.csv')
df2_raw['target'] = df2_raw['Num Vote 2'] / df2_raw['Num Voter']

# 2. รวมข้อมูลดิบทั้งหมดเพื่อแบ่งกลุ่ม
df_all_raw = pd.concat([df1_raw, df2_raw], ignore_index=True)

# 3. แบ่งข้อมูล 9:1 (Train/Val Split) ก่อนทำการสลับฝั่ง
# วิธีนี้จะทำให้ชุด Val ไม่มี "เงา" ของรูปที่เคยเห็นในชุด Train เลย
df_train_raw, df_val = train_test_split(df_all_raw, test_size=VAL_SIZE, random_state=RANDOM_SEED)

# 4. ฟังก์ชันสำหรับการสลับรูป (Augmentation)
def augment_siamese(df):
    df_swapped = df.copy()
    df_swapped['Image 1'], df_swapped['Image 2'] = df['Image 2'], df['Image 1']
    df_swapped['target'] = 1.0 - df['target']
    return pd.concat([df, df_swapped], ignore_index=True)

# 5. สร้างข้อมูลสำหรับแต่ละ Phase
# Phase 1: เฉพาะ Instagram ที่ผ่านการสลับฝั่งแล้ว
df1_train_only, df1_val_only = train_test_split(df1_raw, test_size=VAL_SIZE, random_state=RANDOM_SEED)
train_ds_p1 = SiameseDataset(augment_siamese(df1_train_only).sample(frac=1, random_state=RANDOM_SEED))
val_ds_p1 = SiameseDataset(df1_val_only)

# Phase 2: ข้อมูลผสมทั้งหมด (สลับเฉพาะชุด Train เท่านั้น)
train_ds_p2 = SiameseDataset(augment_siamese(df_train_raw).sample(frac=1, random_state=RANDOM_SEED))
val_ds_p2 = SiameseDataset(df_val)

print(f"Original Unique Pairs: {len(df_all_raw)}")
print(f"Phase 2 Train Size (Augmented): {len(augment_siamese(df_train_raw))}")
print(f"Validation Size (Original): {len(df_val)}")

# ==========================================
# 3. SIAMESE MODEL BUILDING
# ==========================================
def build_model():
    base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1, input_2 = layers.Input(shape=(300, 300, 3)), layers.Input(shape=(300, 300, 3))
    feat_1, feat_2 = base_model(input_1), base_model(input_2)
    
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, mul])
    
    x = layers.Dense(512, activation='swish')(merged)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='swish')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return models.Model(inputs=[input_1, input_2], outputs=output)

keras.backend.clear_session()
model = build_model()

# ==========================================
# 4. TRAINING PHASES
# ==========================================
class MemoryCleanCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        torch.cuda.empty_cache()
        gc.collect()

# Phase 1
model.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy', metrics=['mae'])
print(f"Phase 1: Starting...")
model.fit(train_ds_p1, validation_data=val_ds_p1, epochs=PHASE_1_EPOCHS, callbacks=[MemoryCleanCallback()])

# Phase 2
print("Phase 2: Fine-tuning...")
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=optimizers.Adam(2e-5), loss='binary_crossentropy', metrics=['mae'])

# เปลี่ยนมา monitor 'val_mae' เพื่อป้องกัน Overfit จริงๆ
early_stop = keras.callbacks.EarlyStopping(monitor='val_mae', patience=6, restore_best_weights=True)

model.fit(train_ds_p2, validation_data=val_ds_p2, epochs=PHASE_2_EPOCHS, callbacks=[MemoryCleanCallback(), early_stop])

model.save('food_expert_final.keras')
print("Complete!")