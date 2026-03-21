# Hướng dẫn Training UMPT-SAM — Ablation Study

## 1. Cấu trúc thư mục

```
sam3/                                  # Project root
├── model_trained/
│   └── sam3.pt                        # SAM3 pre-trained checkpoint
├── umpt_sam/
│   ├── data/                          # Đặt dataset tại đây
│   │   ├── kvasir-seg/
│   │   │   ├── images/                # *.jpg hoặc *.png
│   │   │   ├── masks/                 # binary mask *.png (cùng tên với ảnh)
│   │   │   └── split/                 # (tự động tạo nếu chưa có)
│   │   │       ├── train.txt
│   │   │       ├── val.txt
│   │   │       └── test.txt
│   │   ├── CVC-ClinicDB/             # cùng format
│   │   ├── CVC-ColonDB/
│   │   ├── CVC-300/
│   │   └── ETIS-LaribPolypDB/
│   ├── config/
│   ├── losses/
│   ├── modules/
│   └── training/
├── train_ablation.py                  # Entry point training
├── evaluate_ablation.py               # Entry point evaluation
├── run_all_datasets.py                # Chạy tất cả 50 thí nghiệm
└── checkpoints/                       # Kết quả (tự tạo)
```

### Format dataset

Mỗi dataset cần có:
```
dataset_name/
├── images/          # ảnh colonoscopy (.jpg, .png)
├── masks/           # binary mask (.png), cùng tên file với ảnh
└── split/           # (tùy chọn — tự tạo nếu chưa có)
    ├── train.txt    # danh sách tên file (không có extension)
    ├── val.txt
    └── test.txt
```

> **Nếu chưa có file split**, hệ thống sẽ **tự động tạo** với tỷ lệ 80% train / 10% val / 10% test (seed=42, reproducible).

---

## 2. Cài đặt môi trường

```bash
# Tạo môi trường conda
conda create -n py310 python=3.10 -y
conda activate py310

# Cài PyTorch (chọn phiên bản phù hợp CUDA của server)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Cài dependencies
pip install albumentations opencv-python tqdm
```

---

## 3. 10 kịch bản Ablation

| # | Kịch bản | Prompts | MPPG | UPFE | MPCL | Mục đích |
|---|----------|---------|------|------|------|----------|
| 1 | `only_box` | ☑ Box | ✅ | ✅ | ✅ | Ảnh hưởng của từng loại prompt |
| 2 | `only_point` | ☑ Point | ✅ | ✅ | ✅ | |
| 3 | `only_mask` | ☑ Mask | ✅ | ✅ | ✅ | |
| 4 | `only_text` | ☑ Text | ✅ | ✅ | ✅ | |
| 5 | `box_point` | ☑ Box + Point | ✅ | ✅ | ✅ | Tổ hợp 2 prompts |
| 6 | `box_point_mask` | ☑ Box + Point + Mask | ✅ | ✅ | ✅ | Tổ hợp 3 prompts |
| 7 | `wo_mppg` | ☑ All | ❌ | ✅ | ✅ | Vai trò MPPG |
| 8 | `wo_upfe` | ☑ All | ✅ | ❌ | ✅ | Vai trò UPFE |
| 9 | `wo_mpcl` | ☑ All | ✅ | ✅ | ❌ | Vai trò MPCL |
| 10 | `full_model` | ☑ All | ✅ | ✅ | ✅ | Baseline (đầy đủ) |

---

## 4. Lịch trình Training (3-Phase)

| Phase | Tên | Epochs | LR | λ_con | Freeze |
|-------|-----|--------|-----|-------|--------|
| 1 | Warmup | 5 | 1e-4 | 0.0 | IE ✅ PE ✅ MD ✅ |
| 2 | Adaptation | 5 | 5e-5 | 0.0 | IE ✅ PE ❌ MD ✅ |
| 3 | Consistency | 10 | 1e-5 | 0.5 | IE ✅ PE ❌ MD ❌ |

**Tổng: 20 epochs** | **IE** = Image Encoder (luôn frozen), **PE** = Prompt Encoder, **MD** = Mask Decoder

> Với kịch bản `wo_mpcl`: λ_con = 0 ở tất cả phases, K = 0 (không chạy K-perturbation)

---

## 5. Các lệnh Training

### 5.1 Liệt kê tất cả kịch bản

```bash
python train_ablation.py --list
```

### 5.2 Dry-run (kiểm tra nhanh)

```bash
# Chạy thử 1 epoch, 4 samples để xác nhận setup đúng
python train_ablation.py --scenario full_model --dry-run
```

### 5.3 Train 1 kịch bản

```bash
python train_ablation.py \
    --scenario full_model \
    --dataset kvasir_seg \
    --save-dir checkpoints/ablation \
    --device cuda
```

### 5.4 Train tất cả 10 kịch bản

```bash
python train_ablation.py \
    --all \
    --dataset kvasir_seg \
    --save-dir checkpoints/ablation \
    --device cuda
```

### 5.5 Train tất cả kịch bản × tất cả datasets (50 thí nghiệm)

```bash
python run_all_datasets.py \
    --save-dir checkpoints/ablation \
    --device cuda

# Hoặc chỉ 1 dataset, all scenarios:
python run_all_datasets.py --dataset kvasir_seg

# Hoặc 1 scenario, all datasets:
python run_all_datasets.py --scenario full_model
```

### 5.6 Custom SAM checkpoint

```bash
python train_ablation.py \
    --scenario full_model \
    --sam-checkpoint /path/to/custom/sam3.pt
```

---

## 6. Kết quả đầu ra

Sau khi training xong, cấu trúc thư mục kết quả:

```
checkpoints/ablation/
└── kvasir_seg/
    ├── full_model/
    │   └── run_20260321_150000/
    │       ├── best_model.pth          # Checkpoint tốt nhất
    │       ├── latest_model.pth        # Checkpoint cuối cùng
    │       ├── experiment_config.json  # Config kịch bản
    │       ├── training_history.json   # Loss + metrics mỗi epoch
    │       ├── test_results.json       # Metrics đánh giá cuối
    │       └── training_log.txt        # Log chi tiết
    ├── only_box/
    │   └── run_.../
    ├── wo_mppg/
    │   └── run_.../
    └── ...
```

---

## 7. Evaluation

### 7.1 Đánh giá 1 kịch bản

```bash
python evaluate_ablation.py \
    --scenario full_model \
    --checkpoint checkpoints/ablation/kvasir_seg/full_model/run_.../best_model.pth \
    --dataset kvasir_seg
```

### 7.2 Đánh giá tất cả (tự tìm best_model.pth)

```bash
python evaluate_ablation.py \
    --all \
    --base-dir checkpoints/ablation/kvasir_seg
```

### 7.3 Chỉ in bảng tổng hợp (không re-evaluate)

```bash
python evaluate_ablation.py \
    --summary \
    --base-dir checkpoints/ablation/kvasir_seg
```

Kết quả: file `ablation_summary.csv` + bảng Markdown in ra console.

---

## 8. Metrics đánh giá

| Metric | Mô tả |
|--------|--------|
| **Dice** | Overlap giữa prediction và ground truth |
| **mIoU** | Mean Intersection over Union |
| **Precision** | Tỷ lệ đúng trong các pixel dự đoán positive |
| **Recall** | Tỷ lệ phát hiện được polyp thật |
| **F2-score** | F-score thiên về recall (β=2) |
| **Mask AP** | Average Precision cho mask segmentation |

---

## 9. Thay đổi Hyperparameters

Chỉnh sửa trong file `umpt_sam/config/train_config.py`:

```python
@dataclass
class TrainConfig:
    batch_size: int = 32       # Giảm nếu OOM
    K: int = 3                 # Số lần perturbation cho consistency loss

    phase1 = PhaseConfig(name="warmup",      epochs=5,  lr=1e-4, lambda_con=0.0, ...)
    phase2 = PhaseConfig(name="adaptation",  epochs=5,  lr=5e-5, lambda_con=0.0, ...)
    phase3 = PhaseConfig(name="consistency", epochs=10, lr=1e-5, lambda_con=0.5, ...)

    loss_weights = {
        "consistency_loss_weight": 1.0,
        "regularization_loss_weight": 1.0,
        "dice_loss_weight": 1.0,
    }
```

---

## 10. Xử lý sự cố

| Vấn đề | Giải pháp |
|--------|-----------|
| `CUDA out of memory` | Giảm `batch_size` trong `TrainConfig` hoặc dùng `--dry-run` trước |
| `Dataset not found` | Kiểm tra data đặt đúng tại `umpt_sam/data/<tên dataset>/images/` |
| `SAM3 checkpoint not found` | Đảm bảo `model_trained/sam3.pt` tồn tại, hoặc dùng `--sam-checkpoint` |
| `Split files missing` | Tự động tạo, không cần lo. Hoặc tạo thủ công bằng `python -m umpt_sam.data.make_split` |
| `ImportError: sam3.model...` | Đảm bảo chạy từ thư mục project root (`sam3/`) |

---

## 11. Quick Start (Tóm tắt)

```bash
# 1. Activate env
conda activate py310

# 2. Đảm bảo data + checkpoint đúng chỗ
ls umpt_sam/data/kvasir-seg/images/    # Phải có ảnh
ls model_trained/sam3.pt               # Phải có checkpoint

# 3. Dry-run kiểm tra
python train_ablation.py --scenario full_model --dry-run

# 4. Train thực tế
python train_ablation.py --all --dataset kvasir_seg --save-dir checkpoints/ablation

# 5. Xem kết quả
python evaluate_ablation.py --summary --base-dir checkpoints/ablation/kvasir_seg
```
