import cv2
import torch
import os

# 1. เช็ค OpenCV CUDA Support
# (หมายเหตุ: OpenCV ที่ลงผ่าน pip ปกติมักจะไม่มี CUDA ต้อง build เอง 
# แต่รันเช็คไว้ไม่เสียหายครับ)
count = cv2.cuda.getCudaEnabledDeviceCount()
print(f"OpenCV CUDA devices: {count}")

# 2. เช็ค GPU Memory Usage โดยใช้ PyTorch
if torch.cuda.is_available():
    # สั่งให้ PyTorch จอง memory เบื้องต้น (ถ้ามี)
    device = torch.device("cuda")
    
    # ดึงข้อมูล Memory
    # current_allocated: คือ memory ที่ตัวแปรในโค้ดเราใช้งานจริง
    # current_reserved: คือ memory ที่ PyTorch จองไว้จากระบบ (Cache)
    allocated = torch.cuda.memory_allocated(0) / 1e6
    reserved = torch.cuda.memory_reserved(0) / 1e6
    max_mem = torch.cuda.get_device_properties(0).total_memory / 1e6

    print(f"--- GPU: {torch.cuda.get_device_name(0)} ---")
    print(f"Total GPU Memory: {max_mem:.2f} MB")
    print(f"Current Allocated: {allocated:.2f} MB")
    print(f"Current Reserved (Cache): {reserved:.2f} MB")
    print(f"Free Memory (Approx): {max_mem - reserved:.2f} MB")
else:
    print("CUDA is not available.")
    