import torch
import time

# เช็คอุปกรณ์ก่อน
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

def test_operation():
    # สร้าง Matrix ขนาด 10000x10000 เพื่อให้เห็นความต่างชัดขึ้น
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    # Matrix Multiplication
    c = torch.matmul(a, b)
    return c

# 1. ทดสอบแบบ Normal (Eager Mode)
print("\n--- Running Normal Mode ---")
torch.cuda.synchronize() # รอให้ GPU เคลียร์งานเก่า
start = time.time()
test_operation()
torch.cuda.synchronize() # รอให้คำนวณเสร็จจริง
print(f"Normal execution took: {time.time() - start:.4f} seconds")

# 2. ทดสอบแบบ JIT Compilation (torch.compile)
# นี่คือส่วนที่จะใช้เวลา 'Compile' ครั้งแรกเพื่อรีดประสิทธิภาพให้เข้ากับ Blackwell (sm_120)
print("\n--- Starting JIT Compilation (torch.compile) ---")
compiled_test = torch.compile(test_operation)

start_compile = time.time()
compiled_test() # การรันครั้งแรกจะเป็นการ Compile + Run
torch.cuda.synchronize()
print(f"First run (Compile + Execution) took: {time.time() - start_compile:.4f} seconds")

# 3. ทดสอบความเร็วหลัง Compile เสร็จแล้ว
print("\n--- Running Compiled Mode (Post-JIT) ---")
start_post = time.time()
compiled_test()
torch.cuda.synchronize()
print(f"Post-JIT execution took: {time.time() - start_post:.4f} seconds")