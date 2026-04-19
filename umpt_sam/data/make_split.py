import os
import random
from pathlib import Path

# Resolve dataset path relative to this file's location (umpt_sam/data/)
_DATA_DIR = Path(__file__).resolve().parent
DATASET_ROOT = str(_DATA_DIR / "kvasir-sessile" / "sessile-main-Kvasir-SEG")

def create_splits(root_dir, train_ratio=0.8, val_ratio=0.1):
    image_dir = os.path.join(root_dir, "images")
    split_dir = os.path.join(root_dir, "split")
    
    os.makedirs(split_dir, exist_ok=True)
    
    all_files = []
    for f in os.listdir(image_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(f)[0]
            all_files.append(name)
            
    random.seed(42) 
    random.shuffle(all_files)
    
    total = len(all_files)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_files = all_files[:train_end]
    val_files = all_files[train_end:val_end]
    test_files = all_files[val_end:]
    
    # 4. Ghi ra file .txt
    splits = {"train.txt": train_files, "val.txt": val_files, "test.txt": test_files}
    
    for filename, file_list in splits.items():
        filepath = os.path.join(split_dir, filename)
        with open(filepath, "w") as f:
            f.write("\n".join(file_list))
            
    print(f"🎉 Đã chia xong tổng cộng {total} ảnh:")
    print(f"   - Tập Train: {len(train_files)} ảnh")
    print(f"   - Tập Val  : {len(val_files)} ảnh")
    print(f"   - Tập Test : {len(test_files)} ảnh")
    print(f"💾 File được lưu tại: {split_dir}")

if __name__ == "__main__":
    create_splits(DATASET_ROOT, train_ratio=0.8, val_ratio=0.1)