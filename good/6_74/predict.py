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
# รายชื่อไฟล์โมเดลทั้ง 5 Folds
MODEL_PATHS = [f'fold_{i}_v2s_final.keras' for i in range(1, 6)]
TEST_CSV = 'hidden_test.csv'
TEST_IMAGE_FOLDER = 'hidden_images'

# --- 2. Build โครงสร้าง (ต้องตรงกับไฟล์เทรนล่าสุด 1024/512 + L2) ---
def build_model():
    base_model = applications.EfficientNetV2S(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1 = layers.Input(shape=(300, 300, 3))
    input_2 = layers.Input(shape=(300, 300, 3))
    
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    shared_bn = layers.BatchNormalization()
    feat_1 = shared_bn(feat_1)
    feat_2 = shared_bn(feat_2)
    
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    sq_diff = layers.Lambda(lambda x: ops.square(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, sq_diff, mul])
    
    # Dense Layers (1024 -> 512)
    x = layers.Dense(1024, activation='swish')(merged)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# --- 3. โหลดโมเดลทั้งหมดเตรียมไว้ ---
loaded_models = []
print("🔄 Loading All 5 Folds for Ensemble...")
for path in MODEL_PATHS:
    if os.path.exists(path):
        try:
            m = build_model()
            m.load_weights(path)
            loaded_models.append(m)
            print(f"✅ Loaded: {path}")
        except Exception as e:
            print(f"❌ Error loading {path}: {e}")
    else:
        print(f"⚠️ Warning: {path} not found. Skipping.")

if not loaded_models:
    print("❌ No models loaded. Check your filenames!")
    sys.exit()

# --- 4. ฟังก์ชันโหลดรูปขนาน ---
def load_and_preprocess(img_name):
    try:
        path = os.path.join(TEST_IMAGE_FOLDER, str(img_name))
        img = utils.load_img(path, target_size=IMG_SIZE)
        img = utils.img_to_array(img)
        return applications.efficientnet_v2.preprocess_input(img)
    except:
        return np.zeros((*IMG_SIZE, 3))

# --- 5. เริ่มทำนายผลแบบ Ensemble + Symmetry ---
executor = ThreadPoolExecutor(max_workers=16)
df_test = pd.read_csv(TEST_CSV)
print(f"🔮 Predicting {len(df_test)} pairs with 5-Fold Ensemble...")

winners = []
confidences = []

for index, row in df_test.iterrows():
    imgs = list(executor.map(load_and_preprocess, [row['Image 1'], row['Image 2']]))
    img1, img2 = np.expand_dims(imgs[0], axis=0), np.expand_dims(imgs[1], axis=0)
    
    all_fold_probs = []
    
    # วนลูปทำนายด้วยทุกโมเดล
    for model in loaded_models:
        p_norm = float(model.predict([img1, img2], verbose=0)[0][0])
        p_swap = float(model.predict([img2, img1], verbose=0)[0][0])
        # เฉลี่ยสมมาตรของแต่ละ Fold
        fold_prob = (p_norm + (1.0 - p_swap)) / 2.0
        all_fold_probs.append(fold_prob)
    
    # เฉลี่ยรวมทุก Folds (Ensemble)
    final_prob = np.mean(all_fold_probs)
    
    if final_prob < 0.5:
        winner = 1
        confidence = ((0.5 - final_prob) / 0.5) * 100
    else:
        winner = 2
        confidence = ((final_prob - 0.5) / 0.5) * 100
            
    winners.append(winner)
    confidences.append(min(max(confidence, 0), 100))
    
    print(f"[{index+1:03d}/{len(df_test)}] Winner: {winner} | Confidence: {confidences[-1]:.2f}%")

df_test['Winner'] = winners
df_test['Confidence_Percent'] = confidences
df_test.to_csv('final_submission.csv', index=False)
print(f"\n✨ Done! Results saved to 'final_submission.csv'")