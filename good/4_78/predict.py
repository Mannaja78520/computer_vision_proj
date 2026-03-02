import os
import sys
from concurrent.futures import ThreadPoolExecutor

# --- 1. ตั้งค่า Environment ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import torch
import keras
from keras import layers, models, applications, ops, utils

# การตั้งค่า
IMG_SIZE = (300, 300)
MODEL_PATH = 'food_expert_v2s_symmetric.keras' 
TEST_CSV = 'hidden_test.csv'
TEST_IMAGE_FOLDER = 'hidden_images'

# --- 2. Build โครงสร้าง (ต้องเหมือนตอนเทรนเป๊ะๆ) ---
def build_model():
    # ใช้ V2S ตามไฟล์เทรนล่าสุด
    base_model = applications.EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1 = layers.Input(shape=(300, 300, 3))
    input_2 = layers.Input(shape=(300, 300, 3))
    
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    # Shared BatchNormalization
    shared_bn = layers.BatchNormalization()
    feat_1 = shared_bn(feat_1)
    feat_2 = shared_bn(feat_2)
    
    # Interaction Layers (ต้องมี sq_diff ตามที่เทรนไว้)
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    sq_diff = layers.Lambda(lambda x: ops.square(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, sq_diff, mul])
    
    x = layers.Dense(512, activation='swish')(merged)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# --- 3. โหลด Weights ---
print(f"🔄 Loading Model: {MODEL_PATH}...")
try:
    model = build_model()
    model.load_weights(MODEL_PATH)
    print("✅ Weights loaded successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit()

# --- 4. ฟังก์ชันโหลดรูปขนาน ---
def load_and_preprocess(img_name):
    try:
        path = os.path.join(TEST_IMAGE_FOLDER, str(img_name))
        img = utils.load_img(path, target_size=IMG_SIZE)
        img = utils.img_to_array(img)
        # ต้องใช้ preprocess_input ของ V2
        return applications.efficientnet_v2.preprocess_input(img)
    except:
        return np.zeros((*IMG_SIZE, 3))

# --- 5. เริ่มทำนายผล ---
executor = ThreadPoolExecutor(max_workers=16)
df_test = pd.read_csv(TEST_CSV)
print(f"🔮 Predicting {len(df_test)} pairs with V2S Symmetry Engine...")

winners = []
confidences = []

for index, row in df_test.iterrows():
    imgs = list(executor.map(load_and_preprocess, [row['Image 1'], row['Image 2']]))
    img1, img2 = np.expand_dims(imgs[0], axis=0), np.expand_dims(imgs[1], axis=0)
    
    # ทาย 2 ทิศทาง (A,B และ B,A)
    p_norm = float(model.predict([img1, img2], verbose=0)[0][0])
    p_swap = float(model.predict([img2, img1], verbose=0)[0][0])
    
    # เฉลี่ยค่าแบบสมมาตร
    final_prob = (p_norm + (1.0 - p_swap)) / 2.0
    
    if final_prob < 0.5:
        winner = 1
        confidence = ((0.5 - final_prob) / 0.5) * 100
    else:
        winner = 2
        confidence = ((final_prob - 0.5) / 0.5) * 100
            
    winners.append(winner)
    confidences.append(min(max(confidence, 0), 100))
    
    if (index + 1) % 1 == 0 or (index + 1) == len(df_test):
        print(f"[{index+1:03d}/{len(df_test)}] Winner: {winner} | Confidence: {confidences[-1]:.2f}%")

df_test['Winner'] = winners
df_test['Confidence_Percent'] = confidences
df_test.to_csv('final_submission.csv', index=False)
print(f"\n✨ Done! Results saved to 'final_submission.csv'")