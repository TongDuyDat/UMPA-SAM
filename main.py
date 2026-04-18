import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from dataclasses import dataclass

from umpt_sam.umpa_model import UMPAModel
from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.losses.composer_loss import ComposerLoss
from umpt_sam.training.phase_scheduler import PhaseScheduler
from umpt_sam.training.trainer import UMPATrainer
from umpt_sam.training.evaluate import evaluate

from umpt_sam.data.polyp_dataset import PolypDataset 
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
    
from umpt_sam.data.kvasir_sessile import DATASET_SOURCE, TRANSFORM_PIPELINE


def main():
    print("🚀 KHỞI ĐỘNG HỆ THỐNG UMPA-SAM...")
    config = TrainConfig()
    device = "cuda"
    print(f"🖥️ Đang chạy trên thiết bị: {device.upper()}")

    # 1. KHỞI TẠO DATASET & DATALOADER 
    print("📦 Đang load dữ liệu...")
    dataset_config = DatasetConfig.kvasir_sessile(root=DATASET_SOURCE) 
    
    train_dataset = PolypDataset(cfg=dataset_config, phase='train')
    train_dataset.transform = TRANSFORM_PIPELINE['train']  
    
    val_dataset = PolypDataset(cfg=dataset_config, phase='val')
    val_dataset.transform = TRANSFORM_PIPELINE['val']     
    
    test_dataset = PolypDataset(cfg=dataset_config, phase='test')
    test_dataset.transform = TRANSFORM_PIPELINE['val']



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
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    print(f" Đã nạp: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")
    
    print("Đang khởi tạo mô hình UMPA-SAM...")
    model_config = UMPAModelConfig(
        sam_checkpoint="model_trained/sam3.pt", 
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256),
        mppg=MPPGConfig()
    )
    model = UMPAModel.from_config(model_config=model_config).to(device)

    print("Đang thiết lập Loss và Optimizer...")
    composer_loss = ComposerLoss(config_loss=config.loss_weights).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config.phase1.lr, weight_decay=1e-4)
    
    scheduler = PhaseScheduler(train_config=config)

    print("BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN...")
    trainer = UMPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=config,
        composer_loss=composer_loss,
        evaluate_fn=evaluate,  
        save_dir="checkpoints",
        device="cuda"
        )


    trainer.run()
    print("QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT!")

if __name__ == "__main__":
    main()