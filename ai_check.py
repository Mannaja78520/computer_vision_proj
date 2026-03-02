import pandas as pd

# 1. โหลดเฉลย และ ผลที่ AI ทาย
truth = pd.read_csv('hidden_ground_truth.csv')
predict = pd.read_csv('final_submission.csv')

print("คอลัมน์ในไฟล์เฉลย:", truth.columns.tolist())

# 2. ตรวจสอบเงื่อนไข (เลือกใช้ตามชื่อคอลัมน์ที่มีจริงในไฟล์)
if 'target' in truth.columns:
    truth['Winner_Actual'] = [1 if x < 0.5 else 2 for x in truth['target']]
elif 'Winner' in truth.columns:
    truth['Winner_Actual'] = truth['Winner']
else:
    # ถ้าไม่มีทั้งคู่ ลองเดาว่าคอลัมน์สุดท้ายคือคำตอบ
    last_col = truth.columns[-1]
    print(f"ไม่เจอคอลัมน์มาตรฐาน ใช้คอลัมน์ '{last_col}' เป็นเฉลยแทน")
    truth['Winner_Actual'] = [1 if x < 0.5 else 2 for x in truth[last_col]]

# 3. คำนวณ Accuracy
# ตรวจสอบให้แน่ใจว่าจำนวนแถวเท่ากัน
if len(truth) != len(predict):
    print(f"⚠️ คำเตือน: จำนวนข้อมูลไม่เท่ากัน (เฉลย: {len(truth)}, ทาย: {len(predict)})")

correct = (truth['Winner_Actual'] == predict['Winner']).sum()
total = len(truth)
accuracy = (correct / total) * 100

print("-" * 30)
print(f"AI ทายถูกทั้งหมด: {correct} จาก {total} คู่")
print(f"คิดเป็นความแม่นยำ: {accuracy:.2f}%")