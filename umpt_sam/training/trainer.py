# umpt_sam/training/trainer.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_one_epoch(model, train_loader, optimizer, composer_loss, epoch, phase, K, device="cuda"):
    model.train()
    
    epoch_total_loss = 0.0
    epoch_seg_loss = 0.0
    epoch_con_loss = 0.0

    # 1. Cập nhật lambda_con từ Phase hiện tại vào thẳng ComposerLoss
    composer_loss.config_loss['consistency_loss_weight'] = phase.lambda_con

    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch} [{getattr(phase, 'name', f'Phase {epoch}')}]")
    
    for batch in pbar:
        # Bóc tách dữ liệu
        images = batch['image'].to(device)
        gt_masks = batch['mask'].to(device)  # Đảm bảo key này khớp với dataset của sếp
        
        boxes = batch.get('bbox', None)
        points = batch.get('points', None)
        point_labels = batch.get('point_labels', None)
        captions = batch.get('text', None)
        
        if boxes is not None: boxes = boxes.to(device)
        if points is not None: points = points.to(device)
        if point_labels is not None: point_labels = point_labels.to(device)

        optimizer.zero_grad()

        # ---------------------------------------------------------
        # BƯỚC 1: FORWARD LUỒNG CHÍNH ĐỂ LẤY PRED_MASKS
        # ---------------------------------------------------------
        outputs = model(
            image=images,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            captions=captions
        )
        pred_masks = outputs['pred_masks'] # Shape đang là (B, 1, 288, 288)
        
        # Phóng to mask dự đoán lên bằng kích thước Ground Truth (VD: 1008x1008)
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)

        # ---------------------------------------------------------
        # BƯỚC 2: FORWARD K LUỒNG NHIỄU (Chỉ chạy khi cần thiết)
        # ---------------------------------------------------------
        perturbed_masks = None
        # Chỉ tốn thời gian sinh K-mask nếu K > 0 và lambda_con > 0 (Tức là Phase 3)
        if K > 0 and phase.lambda_con > 0.0:
            k_outputs = model.forward_k_perturbations(
                image=images,
                K=K,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                captions=captions
            )
            
            # Lấy pred_masks từ K outputs và phóng to chúng lên
            p_masks_list = []
            for out in k_outputs:
                p_mask = out['pred_masks']
                if p_mask.shape[-2:] != gt_masks.shape[-2:]:
                    p_mask = F.interpolate(p_mask, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)
                p_masks_list.append(p_mask)
                
            # Gộp lại thành shape chuẩn cho Consistency Loss: (B, K, 1, H, W)
            perturbed_masks = torch.stack(p_masks_list, dim=1)

        # ---------------------------------------------------------
        # BƯỚC 3: TÍNH LOSS
        # ---------------------------------------------------------
        loss_dict = composer_loss(
            pred_masks=pred_masks, 
            gt_masks=gt_masks, 
            perturbed_masks=perturbed_masks
        )
        total_loss = loss_dict['total_loss']

        # ---------------------------------------------------------
        # BƯỚC 4: BACKPROPAGATION
        # ---------------------------------------------------------
        total_loss.backward()
        optimizer.step()

        # Log dữ liệu
        epoch_total_loss += total_loss.item()
        epoch_seg_loss += loss_dict['seg_loss'].item()
        # Nếu lambda_con = 0 thì consistency_loss có thể là float 0.0 hoặc tensor
        con_loss_val = loss_dict['consistency_loss'].item() if isinstance(loss_dict['consistency_loss'], torch.Tensor) else 0.0
        epoch_con_loss += con_loss_val

        pbar.set_postfix({
            "L_tot": f"{total_loss.item():.4f}", 
            "L_seg": f"{loss_dict['seg_loss'].item():.4f}",
            "L_con": f"{con_loss_val:.4f}"
        })

    num_batches = len(train_loader)
    return {
        "total_loss": epoch_total_loss / num_batches,
        "seg_loss": epoch_seg_loss / num_batches,
        "consistency_loss": epoch_con_loss / num_batches
    }


def train(model, train_loader, val_loader, optimizer, scheduler, train_config, composer_loss, evaluate_fn, save_dir="checkpoints", device="cuda"):
    """
    Hàm Train chính. Nhận evaluate_fn từ bên ngoài vào.
    """
    os.makedirs(save_dir, exist_ok=True)
    best_val_dice = 0.0
    total_epochs = scheduler.total_epochs

    for epoch in range(1, total_epochs + 1):
        phase = scheduler.get_current_phase(epoch)
        scheduler.apply_phase(model=model, epoch=epoch, optimizer=optimizer)
        
        print(f"\n=== Epoch {epoch}/{total_epochs} | Phase: {getattr(phase, 'name', 'Phase')} "
              f"| LR: {phase.lr} | lambda_con: {phase.lambda_con} ===")

        # Chạy 1 epoch train
        train_info = train_one_epoch(
            model=model, 
            train_loader=train_loader, 
            optimizer=optimizer, 
            composer_loss=composer_loss,  
            epoch=epoch, 
            phase=phase, 
            K=train_config.K, 
            device=device
        )

        # GỌI HÀM EVALUATE TỪ BÊN NGOÀI TRUYỀN VÀO
        val_metrics = evaluate_fn(model, val_loader, device=device)

        print(f"[Log] Epoch {epoch} | Phase {getattr(phase, 'name', 'Phase')} "
              f"| L_seg: {train_info['seg_loss']:.4f} "
              f"| L_con: {train_info['consistency_loss']:.4f} "
              f"| L_total: {train_info['total_loss']:.4f} "
              f"| val_dice: {val_metrics.get('dice', 0.0):.4f}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_metrics.get('dice', 0.0)
        }
        
        torch.save(checkpoint, os.path.join(save_dir, "latest_model.pth"))
        
        if val_metrics.get('dice', 0.0) > best_val_dice:
            best_val_dice = val_metrics.get('dice', 0.0)
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))
            print(f"🔥 Checkpoint ĐỈNH NHẤT đã được lưu với val_dice: {best_val_dice:.4f}")