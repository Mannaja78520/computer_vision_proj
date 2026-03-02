import os
import sys
from concurrent.futures import ThreadPoolExecutor # เพิ่มตัวขนานงาน CPU

# --- 1. ตั้งค่า Backend ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import torch
import keras
from keras import layers, models, applications, ops, utils
from sklearn.model_selection import train_test_split

# การตั้งค่า
IMG_SIZE = (300, 300)
BATCH_SIZE = 6 
MODEL_PATH = 'food_expert_final.keras'
TEST_CSV = 'hidden_test.csv'
TEST_IMAGE_FOLDER = 'hidden_images'

# --- 2. สร้าง Model ให้ตรงกับตอนเทรน (Symmetric Architecture) ---
def build_siamese_model():
    base_model = applications.EfficientNetB3(weights=None, include_top=False, pooling='avg')
    
    input_1 = layers.Input(shape=(300, 300, 3))
    input_2 = layers.Input(shape=(300, 300, 3))
    
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    # คำนวณความต่างแบบเดียวกับตอนเทรน
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    # ห้ามใส่ add เพราะตอนเทรนเราไม่ได้ใส่
    
    # รวม 4 ตัวให้ตรงกับค่า Received: 6144 ใน Error message
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, mul])
    
    x = layers.Dense(512, activation='swish')(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation='swish')(x)
    # ใส่ dtype='float32' ให้เหมือนตอนเทรนเพื่อความชัวร์
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    
    return models.Model(inputs=[input_1, input_2], outputs=output)

# --- 3. โหลดโมเดล ---
model = build_siamese_model()
if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)
    print("Weights loaded successfully!")

# --- 4. ฟังก์ชันโหลดรูปแบบขนาน (Parallel Image Loading) ---
def load_and_preprocess(img_name):
    try:
        path = os.path.join(TEST_IMAGE_FOLDER, str(img_name))
        img = utils.load_img(path, target_size=IMG_SIZE)
        img = utils.img_to_array(img)
        return applications.efficientnet.preprocess_input(img)
    except Exception as e:
        return np.zeros((*IMG_SIZE, 3))

# ใช้ ThreadPoolExecutor ดึงพลังจาก 16-20 คอร์
executor = ThreadPoolExecutor(max_workers=24)

# --- 5. เริ่มทำนายผล ---
df_test = pd.read_csv(TEST_CSV)
print(f"Predicting {len(df_test)} pairs using Multi-core CPU Loading...")

winners = []
confidences = []

for index, row in df_test.iterrows():
    # โหลด 2 รูปพร้อมกันคนละ Thread
    imgs = list(executor.map(load_and_preprocess, [row['Image 1'], row['Image 2']]))
    img1, img2 = np.expand_dims(imgs[0], axis=0), np.expand_dims(imgs[1], axis=0)
    
    output = model.predict([img1, img2], verbose=0)
    prob = float(output[0][0])
    
    # Logic: 0.05 คือ Winner 1 และ 0.95 คือ Winner 2
    if prob < 0.5:
        winner = 1
        confidence = (1 - prob) * 100
    else:
        winner = 2
        confidence = prob * 100
            
    winners.append(winner)
    confidences.append(confidence)
    print(f"[{index+1:03d}] Winner: {winner} | Confidence: {confidence:.2f}%")

df_test['Winner'] = winners
df_test['Confidence_Percent'] = confidences
df_test.to_csv('final_submission.csv', index=False)
print("\nDone!")