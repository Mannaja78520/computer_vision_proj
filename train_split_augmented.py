import pandas as pd

# 1. โหลดข้อมูลที่คุณส่งมา (สมมติว่าชื่อไฟล์เดิมคือ train_split.csv)
df = pd.read_csv('train_split.csv')

# 2. สร้างชุดข้อมูล "สลับฝั่ง" (Swapped Set)
df_swapped = df.copy()

# สลับชื่อไฟล์รูปภาพ
df_swapped['Image 1'] = df['Image 2']
df_swapped['Image 2'] = df['Image 1']

# สลับค่า Winner (ถ้าเดิมเป็น 1 ให้เป็น 2, ถ้าเป็น 2 ให้เป็น 1)
# วิธีนี้ใช้สำหรับ Winner ที่เป็นเลข 1 หรือ 2 เท่านั้น
df_swapped['Winner'] = df['Winner'].map({1: 2, 2: 1})

# สำหรับค่า target (ถ้ามี) สลับจาก 0.0 เป็น 1.0 หรือ 1.0 เป็น 0.0
if 'target' in df_swapped.columns:
    df_swapped['target'] = 1.0 - df['target']

# 3. นำข้อมูลเดิมมารวมกับข้อมูลที่สลับแล้ว
df_augmented = pd.concat([df, df_swapped], ignore_index=True)

# 4. Shuffle ข้อมูลเพื่อให้คละกัน
df_augmented = df_augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. บันทึกเป็นไฟล์ใหม่
df_augmented.to_csv('train_split_augmented.csv', index=False)

print(f"Original data: {len(df)} pairs")
print(f"Augmented data (Swapped): {len(df_augmented)} pairs")
print("สร้างไฟล์ 'train_split_augmented.csv' เรียบร้อยแล้ว!")