import os
import sys
from concurrent.futures import ThreadPoolExecutor

# --- 1. SET ENVIRONMENT ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
import torch
import keras
from keras import layers, models, applications, ops, utils

# --- CONFIGURATION ---
IMG_SIZE = (300, 300)
TEST_CSV = 'hidden_test.csv'
TEST_IMAGE_FOLDER = 'hidden_images'

# รายชื่อโมเดลและประเภท (ต้องตรงกับที่เทรนไว้)
MODEL_CONFIGS = [
    {'path': 'fold_1_v2s_final.keras', 'type': 'v2s'},
    {'path': 'fold_2_v2s_final.keras', 'type': 'v2s'},
    {'path': 'fold_3_convnext_final.keras', 'type': 'convnext'},
    {'path': 'fold_4_convnext_final.keras', 'type': 'convnext'},
    {'path': 'fold_5_v2m_final.keras', 'type': 'v2m'}
]

# ==========================================
# 2. MODEL BUILDER (ต้องตรงกับไฟล์เทรน)
# ==========================================
def build_model_architecture(model_type):
    if model_type == 'v2s':
        base = applications.EfficientNetV2S(weights=None, include_top=False, pooling='avg')
    elif model_type == 'convnext':
        base = applications.ConvNeXtTiny(weights=None, include_top=False, pooling='avg')
    elif model_type == 'v2m':
        base = applications.EfficientNetV2M(weights=None, include_top=False, pooling='avg')
    
    input_1 = layers.Input(shape=(*IMG_SIZE, 3))
    input_2 = layers.Input(shape=(*IMG_SIZE, 3))
    feat_1, feat_2 = base(input_1), base(input_2)
    
    shared_bn = layers.BatchNormalization()
    feat_1, feat_2 = shared_bn(feat_1), shared_bn(feat_2)
    
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    sq_diff = layers.Lambda(lambda x: ops.square(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, sq_diff, mul])
    
    x = layers.Dense(1024, activation='swish')(merged)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.4)(x)
    x = layers.Dense(512, activation='swish')(x)
    x = layers.BatchNormalization()(x); x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)

# ==========================================
# 3. PREPROCESSING LOGIC
# ==========================================
def preprocess_image(img, model_type):
    if model_type == 'v2s' or model_type == 'v2m':
        return applications.efficientnet_v2.preprocess_input(img)
    elif model_type == 'convnext':
        return applications.convnext.preprocess_input(img)
    return img / 255.0

def load_raw_image(img_name):
    try:
        path = os.path.join(TEST_IMAGE_FOLDER, str(img_name))
        img = utils.load_img(path, target_size=IMG_SIZE)
        return utils.img_to_array(img)
    except:
        return np.zeros((*IMG_SIZE, 3))

# ==========================================
# 4. LOAD MODELS
# ==========================================
loaded_models = []
print("🔄 Loading Hybrid Ensemble Models...")
for cfg in MODEL_CONFIGS:
    if os.path.exists(cfg['path']):
        m = build_model_architecture(cfg['type'])
        m.load_weights(cfg['path'])
        loaded_models.append({'model': m, 'type': cfg['type']})
        print(f"✅ Loaded {cfg['type'].upper()}: {cfg['path']}")
    else:
        print(f"⚠️ Missing: {cfg['path']}")

if not loaded_models:
    print("❌ No models found!")
    sys.exit()

# ==========================================
# 5. PREDICTION LOOP
# ==========================================
df_test = pd.read_csv(TEST_CSV)
executor = ThreadPoolExecutor(max_workers=16)
winners, confidences = [], []

print(f"🔮 Predicting {len(df_test)} pairs with Diverse Hybrid Intelligence...")

for index, row in df_test.iterrows():
    # โหลดรูปดิบครั้งเดียว
    raw_imgs = list(executor.map(load_raw_image, [row['Image 1'], row['Image 2']]))
    
    fold_probs = []
    for m_info in loaded_models:
        # Preprocess ให้ตรงกับสายพันธุ์ของโมเดลนั้นๆ
        img1 = preprocess_image(raw_imgs[0].copy(), m_info['type'])
        img2 = preprocess_image(raw_imgs[1].copy(), m_info['type'])
        
        batch_norm = [np.expand_dims(img1, 0), np.expand_dims(img2, 0)]
        batch_swap = [np.expand_dims(img2, 0), np.expand_dims(img1, 0)]
        
        # Symmetry Prediction
        p_norm = float(m_info['model'].predict(batch_norm, verbose=0)[0][0])
        p_swap = float(m_info['model'].predict(batch_swap, verbose=0)[0][0])
        
        fold_probs.append((p_norm + (1.0 - p_swap)) / 2.0)
    
    # เฉลี่ยผลลัพธ์จากโมเดลทุกสายพันธุ์
    final_prob = np.mean(fold_probs)
    
    if final_prob < 0.5:
        winner, conf = 1, ((0.5 - final_prob) / 0.5) * 100
    else:
        winner, conf = 2, ((final_prob - 0.5) / 0.5) * 100
            
    winners.append(winner)
    confidences.append(min(max(conf, 0), 100))
    print(f"[{index+1:03d}/{len(df_test)}] Winner: {winner} | Confidence: {confidences[-1]:.2f}%")

df_test['Winner'] = winners
df_test['Confidence_Percent'] = confidences
df_test.to_csv('final_submission.csv', index=False)
print(f"\n✨ Done! Results saved to 'final_submission.csv'")