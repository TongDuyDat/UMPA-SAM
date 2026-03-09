# umpt_sam/evaluate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm

@torch.no_grad()  # Tắt tính gradient để tiết kiệm VRAM và tăng tốc
def evaluate(model, val_loader, device="cuda"):
    """
    Hàm đánh giá mô hình trên tập Validation.
    Trả về Dictionary chứa các metric (ví dụ: dice, miou...).
    """
    model.eval()
    total_dice = 0.0
    
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch in pbar:
        # Bóc tách dữ liệu giống hệt bên Train
        images = batch['image'].to(device)
        gt_masks = batch['mask'].to(device) 
        
        boxes = batch.get('bbox', None)
        points = batch.get('points', None)
        point_labels = batch.get('point_labels', None)
        captions = batch.get('text', None)
        
        if boxes is not None: boxes = boxes.to(device)
        if points is not None: points = points.to(device)
        if point_labels is not None: point_labels = point_labels.to(device)

        # Forward model
        outputs = model(
            image=images,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            captions=captions
        )
        pred_masks = outputs['pred_masks']
        
        # BẮT BUỘC Phóng to mask dự đoán lên bằng GT để tính Dice
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)
        
        # Tính Dice Score
        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()
        intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
        
        # Thêm 1e-6 để tránh lỗi chia cho 0
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        
        mean_dice = dice.mean().item()
        total_dice += mean_dice
        
        pbar.set_postfix({"Dice": f"{mean_dice:.4f}"})

    # Có thể mở rộng thêm tính mIoU ở đây và nhét vào dict trả về
    return {
        "dice": total_dice / len(val_loader)
    }