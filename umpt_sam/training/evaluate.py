# umpt_sam/evaluate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sam3.train.loss.loss_fns import segment_miou

def evaluate(model, val_loader, device="cuda"):
    """
    Hàm đánh giá mô hình trên tập Validation.
    Trả về Dictionary chứa các metric (ví dụ: dice, miou...).
    """
    model.eval()
    total_miou, total_dice = 0.0, 0.0
    
    pbar = tqdm(val_loader, desc="Validating")
    
    for batch in pbar:
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
        
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(pred_masks, size=gt_masks.shape[-2:], mode="bilinear", align_corners=False)
        
        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()

        #Tinh MIOU
        pred_3d = pred_binary.squeeze(1).bool()
        gt_3d = (gt_masks.squeeze(1) > 0.5).bool()
        
        batch_miou = segment_miou(pred_3d, gt_3d)
        total_miou += batch_miou.item()



        # Tính Dice Score
        intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
        
        # Thêm 1e-6 để tránh lỗi chia cho 0
        dice = (2. * intersection + 1e-6) / (union + 1e-6)
        
        mean_dice = dice.mean().item()
        total_dice += mean_dice
        
        pbar.set_postfix({
            "Dice": f"{mean_dice:.4f}",
            "mIoU": f"{batch_miou.item():.4f}"
        })


    return {
        "dice": total_dice / len(val_loader),
        "miou": total_miou / len(val_loader)
    }