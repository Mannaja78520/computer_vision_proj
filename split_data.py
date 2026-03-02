import pandas as pd
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

# --- CONFIG ---
TEST_SIZE = 0.10  # ดึงมา 10% ก่อน แล้วค่อย Clone เป็น 20%
RANDOM_SEED = 42
IMAGE_SRC = 'all_train_pic/'
HIDDEN_IMAGE_DIR = 'hidden_images/'

# 1. เตรียมข้อมูล (IG ปรับค่า Smooth, Quest ปรับตามโหวต)
df_insta = pd.read_csv('data_from_intragram.csv')
df_insta['target'] = np.where(df_insta['Winner'] == 1, 0.1, 0.9)
df_insta['weight'] = 1.0

df_quest = pd.read_csv('data_from_questionaire.csv')
df_quest['target'] = df_quest['Num Vote 2'] / df_quest['Num Voter']
df_quest['weight'] = df_quest['Num Voter'] / df_quest['Num Voter'].mean()

# รวมร่าง
cols = ['Image 1', 'Image 2', 'Menu', 'target', 'weight', 'Winner']
df_combined = pd.concat([df_insta[cols], df_quest[cols]], ignore_index=True)

# 2. Split 10% ออกมาเป็นฐานสำหรับ Hidden Test
df_train_full, df_hidden_base = train_test_split(df_combined, test_size=TEST_SIZE, random_state=RANDOM_SEED)

# 3. การ Clone และสลับฝั่ง (Symmetry Cloning)
# สร้างชุดสลับฝั่งจากชุด Hidden Base
df_hidden_swapped = df_hidden_base.copy()
df_hidden_swapped['Image 1'], df_hidden_swapped['Image 2'] = df_hidden_base['Image 2'], df_hidden_base['Image 1']
df_hidden_swapped['Winner'] = df_hidden_base['Winner'].map({1: 2, 2: 1})
df_hidden_swapped['target'] = 1.0 - df_hidden_base['target']

# รวมชุด Original และ Swapped เข้าด้วยกัน (ทำให้ชุด Test ใหญ่ขึ้น 2 เท่า)
df_test_final = pd.concat([df_hidden_base, df_hidden_swapped], ignore_index=True)

# 4. เซฟไฟล์
df_train_full.to_csv('train_split.csv', index=False)
df_test_final.to_csv('hidden_ground_truth.csv', index=False)
df_test_predict = df_test_final.copy()
df_test_predict['Winner'] = 0
df_test_predict.to_csv('hidden_test.csv', index=False)

# 5. Copy รูปภาพ
print("Copying images...")
unique_imgs = pd.concat([df_test_final['Image 1'], df_test_final['Image 2']]).unique()
for img in unique_imgs:
    src = os.path.join(IMAGE_SRC, str(img))
    dst = os.path.join(HIDDEN_IMAGE_DIR, str(img))
    if os.path.exists(src): shutil.copy(src, dst)

print(f"Symmetry Split Complete!")
print(f"Train: {len(df_train_full)} | Hidden Test (Cloned): {len(df_test_final)}")