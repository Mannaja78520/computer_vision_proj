import os

# --- บังคับใช้ Torch Backend และปิดการทำงานของ TensorFlow ที่พัง ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# บังคับการจัดการหน่วยความจำสำหรับ Blackwell GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

import sys
import pandas as pd
import numpy as np
import torch
import keras
import sys
import pandas as pd
import numpy as np
import torch
import keras
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from keras import layers, models, applications, ops, utils

# --- 1. ตั้งค่าพื้นฐาน ---
os.environ["KERAS_BACKEND"] = "torch"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

IMG_SIZE = (300, 300)
MODEL_PATH = 'food_expert_final.keras'
TEST_CSV = 'hidden_test.csv'
TEST_IMAGE_FOLDER = 'hidden_images'
OUTPUT_DIR = 'visual_analysis'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 2. Build Model (Symmetric Architecture) ---
def build_model():
    base_model = applications.EfficientNetB3(weights='imagenet', include_top=False, pooling='avg')
    base_model.trainable = False 
    
    input_1 = layers.Input(shape=(300, 300, 3))
    input_2 = layers.Input(shape=(300, 300, 3))
    
    feat_1 = base_model(input_1)
    feat_2 = base_model(input_2)
    
    # 1. ใส่ BN ตั้งแต่ต้นเพื่อ Normalize ฟีเจอร์จาก B3
    bn_entry = layers.BatchNormalization()
    feat_1 = bn_entry(feat_1)
    feat_2 = bn_entry(feat_2)
    
    # 2. คำนวณความต่างและความสัมพันธ์
    abs_diff = layers.Lambda(lambda x: ops.abs(x[0] - x[1]))([feat_1, feat_2])
    mul = layers.Multiply()([feat_1, feat_2])
    merged = layers.Concatenate()([feat_1, feat_2, abs_diff, mul])
    
    # 3. Dense Block ที่มีความซับซ้อนพอดี (Deep but Regularized)
    x = layers.Dense(512, activation='swish')(merged)
    x = layers.BatchNormalization()(x) # คุมข้อมูลหลัง Dense
    x = layers.Dropout(0.4)(x)         # เพิ่ม Dropout กัน Overfit
    
    x = layers.Dense(128, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    output = layers.Dense(1, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs=[input_1, input_2], outputs=output)


model = build_model()
model.load_weights(MODEL_PATH)
print("Weights loaded successfully!")

# --- 3. Grad-CAM Logic (Torch Backend Compatible) ---
def get_gradcam(img_array, model, layer_name="top_activation"):
    # ดึง Base Model (EfficientNetB3) ออกมา
    # model.layers[2] คือตำแหน่งของ EfficientNet ใน Siamese model ของเรา
    base_model = model.layers[2] 
    
    # สร้างโมเดลย่อยเพื่อดึง Layer ที่เราต้องการทำ Heatmap
    grad_model = models.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])
    
    # รันผ่านโมเดล
    conv_outputs, _ = grad_model(img_array)
    
    # คำนวณค่าเฉลี่ยของ Activation Map
    heatmap = ops.mean(conv_outputs, axis=-1)[0]
    
    # --- จุดที่ต้องแก้: ย้ายจาก GPU กลับมา CPU ก่อนแปลงเป็น Numpy ---
    if hasattr(heatmap, "cpu"):
        heatmap = heatmap.cpu() # ย้ายจาก CUDA มา CPU สำหรับ Torch Backend
        
    heatmap = heatmap.numpy()
    
    # Normalize Heatmap
    heatmap = np.maximum(heatmap, 0)
    denom = (np.max(heatmap) + 1e-10)
    heatmap /= denom
    return heatmap

def get_gradcam_gpu(img_tensor, model, layer_name="top_activation"):
    base_model = model.layers[2]
    grad_model = models.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])
    
    conv_outputs, _ = grad_model(img_tensor)
    
    heatmap = ops.mean(conv_outputs, axis=-1)[0]
    heatmap = ops.maximum(heatmap, 0)
    heatmap = heatmap / (ops.max(heatmap) + 1e-10)
    
    # --- แก้จุดนี้: เพิ่ม .detach() เพื่อตัด Gradient ออกก่อนแปลงเป็น Numpy ---
    if hasattr(heatmap, "cpu"):
        heatmap = heatmap.detach().cpu() # ตัดสายสัมพันธ์ และย้ายมา CPU
    
    return heatmap.numpy()

def save_visual_result(img_path, heatmap, pair_idx, side, menu, winner_label):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    heatmap = cv2.resize(heatmap, IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # ===== ใช้ชื่อเดียวกับไฟล์ที่จะ save =====
    filename = f"Pair{pair_idx+1}_{menu}_{side}_win{winner_label}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)

    # ===== สร้างแถบด้านบน (ไม่ทับภาพ) =====
    header_height = 40
    header = np.zeros((header_height, IMG_SIZE[1], 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    (text_width, text_height), _ = cv2.getTextSize(filename, font, font_scale, thickness)

    text_x = (IMG_SIZE[1] - text_width) // 2
    text_y = (header_height + text_height) // 2 - 5

    cv2.putText(
        header,
        filename,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # ===== ต่อ header + รูปจริง =====
    final_image = np.vstack([header, superimposed_img])

    cv2.imwrite(save_path, final_image)

# --- 4. Prediction Loop ---
def load_img(name):
    try:
        path = os.path.join(TEST_IMAGE_FOLDER, str(name))
        img = utils.load_img(path, target_size=IMG_SIZE)
        return utils.img_to_array(img), applications.efficientnet.preprocess_input(utils.img_to_array(img))
    except:
        return np.zeros((*IMG_SIZE, 3)), np.zeros((*IMG_SIZE, 3))

executor = ThreadPoolExecutor(max_workers=24)
df_test = pd.read_csv(TEST_CSV)
winners, confidences = [], []

print(f"Starting Prediction & Visual Analysis for {len(df_test)} pairs...")

for index, row in df_test.iterrows():
    # โหลดรูป (ได้ทั้งรูปดิบและรูปที่ preprocess แล้ว)
    raw1, proc1 = load_img(row['Image 1'])
    raw2, proc2 = load_img(row['Image 2'])
    
    img1, img2 = np.expand_dims(proc1, axis=0), np.expand_dims(proc2, axis=0)
    output = model.predict([img1, img2], verbose=0)
    prob = float(output[0][0])
    
    winner = 1 if prob < 0.5 else 2
    conf = (1 - prob) * 100 if winner == 1 else prob * 100
    
    # สร้าง Grad-CAM วิเคราะห์ว่า AI มองอะไรในทั้ง 2 รูป
    heatmap1 = get_gradcam_gpu(img1, model)
    heatmap2 = get_gradcam_gpu(img2, model)
    
    # บันทึกภาพลงโฟลเดอร์แยกตามเมนู (Sushi/Ramen/ฯลฯ)
    save_visual_result(os.path.join(TEST_IMAGE_FOLDER, row['Image 1']), heatmap1, index, "Left", row['Menu'], winner)
    save_visual_result(os.path.join(TEST_IMAGE_FOLDER, row['Image 2']), heatmap2, index, "Right", row['Menu'], winner)
    
    winners.append(winner)
    confidences.append(conf)
    print(f"[{index+1:03d}] {row['Menu']} | Winner: {winner} ({conf:.2f}%) | Visual saved.")

df_test['Winner'], df_test['Confidence'] = winners, confidences
df_test.to_csv('final_submission.csv', index=False)
print(f"\nวิเคราะห์เสร็จสิ้น! ตรวจสอบภาพได้ในโฟลเดอร์: {OUTPUT_DIR}")