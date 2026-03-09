import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataclasses import dataclass

# --- IMPORT CÁC MODULE CỦA SẾP ---
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.losses.composer_loss import ComposerLoss
from umpt_sam.training.phase_scheduler import PhaseScheduler
from umpt_sam.training.trainer import train
from umpt_sam.training.evaluate import evaluate

# Import Dataset của sếp (Giả định sếp đã viết xong ở Ngày 2)
from umpt_sam.data.polyp_dataset import PolypDataset 
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
    
    # IMPORT FILE CONFIG MỚI CỦA SẾP (Nhớ đổi tên file thành kvasir_config.py)
from umpt_sam.data.kvasir_sessile import DATASET_SOURCE, TRANSFORM_PIPELINE
# ==========================================
# 1. CẤU HÌNH TRAINING VÀ CHIA PHASE (Ngày 1)
# ==========================================
@dataclass
class PhaseConfig:
    name: str
    epochs: int
    lr: float
    lambda_con: float
    freeze_image_encoder: bool = True
    freeze_prompt_encoder: bool = False
    freeze_mask_decoder: bool = False

@dataclass
class TrainConfig:
    K: int = 3 # Số lượng mask nhiễu
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Cấu hình 3 Phase y hệt PLAN (1).md
    phase1 = PhaseConfig(name="Warm-up",    epochs=1,  lr=1e-4, lambda_con=0.0, freeze_prompt_encoder=True, freeze_mask_decoder=True)
    phase2 = PhaseConfig(name="Adaptation", epochs=1,  lr=5e-5, lambda_con=0.0, freeze_prompt_encoder=False, freeze_mask_decoder=True)
    phase3 = PhaseConfig(name="Consistency",epochs=1, lr=1e-5, lambda_con=0.5, freeze_prompt_encoder=False, freeze_mask_decoder=False)

# ==========================================
# HÀM MAIN CHÍNH
# ==========================================
def main():
    print("🚀 KHỞI ĐỘNG HỆ THỐNG UMPA-SAM...")
    config = TrainConfig()
    
    #SMOKE TEST (Task 7.1)
    # config.phase1.epochs = config.phase2.epochs = config.phase3.epochs = 1
    
    # device = config.device
    device = "cpu"
    print(f"🖥️ Đang chạy trên thiết bị: {device.upper()}")

    # 1. KHỞI TẠO DATASET & DATALOADER 
    print("📦 Đang load dữ liệu...")
    dataset_config = DatasetConfig.kvasir_sessile(root=DATASET_SOURCE) 
    
    train_dataset = PolypDataset(cfg=dataset_config, phase='train')
    train_dataset.transform = TRANSFORM_PIPELINE['train']  
    
    val_dataset = PolypDataset(cfg=dataset_config, phase='val')
    val_dataset.transform = TRANSFORM_PIPELINE['val']     
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    # --- DUMMY LOADER (Chỉ để test code chạy không lỗi) ---
    # class DummyLoader:
    #     def __init__(self, num_batches=5): self.num_batches = num_batches
    #     def __len__(self): return self.num_batches
    #     def __iter__(self):
    #         for _ in range(self.num_batches):
    #             yield {
    #                 'image': torch.randn(config.batch_size, 3, 1024, 1024),
    #                 'mask': torch.randint(0, 2, (config.batch_size, 1, 1008, 1008)).float(),
    #                 'bbox': torch.tensor([[[100, 100, 200, 200]]] * config.batch_size, dtype=torch.float),
    #                 'points': torch.tensor([[[504, 504]]] * config.batch_size, dtype=torch.float),
    #                 'point_labels': torch.tensor([[1]] * config.batch_size)
    #             }
    # train_loader = DummyLoader(num_batches=5)
    # val_loader = DummyLoader(num_batches=2)
    # -----------------------------------------------------

    # 2. KHỞI TẠO MÔ HÌNH UMPA-SAM
    print("🧠 Đang khởi tạo mô hình UMPA-SAM...")
    model_config = UMPAModelConfig(
        sam_checkpoint="sam3.pt", # Thay bằng đường dẫn thật tới pre-trained weights
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256),
        mppg=MPPGConfig()
    )
    model = UMPAModel.from_config(model_config=model_config).to(device)

    # 3. KHỞI TẠO LOSS & SCHEDULER & OPTIMIZER
    print("⚖️ Đang thiết lập Loss và Optimizer...")
    # Khởi tạo loss với trọng số mặc định (sẽ được cập nhật động bởi PhaseScheduler)
    initial_loss_config = {"consistency_loss_weight": 0.0, "regularization_loss_weight": 0.0}
    composer_loss = ComposerLoss(config_loss=initial_loss_config).to(device)

    # Mặc định lấy LR của Phase 1 để gán cho Optimizer trước
    optimizer = optim.AdamW(model.parameters(), lr=config.phase1.lr, weight_decay=1e-4)
    
    # Trình quản lý Phase
    scheduler = PhaseScheduler(train_config=config)

    # 4. BẮT ĐẦU HUẤN LUYỆN
    print("🔥 BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        composer_loss=composer_loss,
        evaluate_fn=evaluate,
        save_dir="checkpoints",
        device=device
    )
    
    print("🎉 QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT!")

if __name__ == "__main__":
    main()