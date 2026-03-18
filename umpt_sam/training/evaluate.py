# umpt_sam/evaluate.py
import time
from dataclasses import replace
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.train.loss.loss_fns import segment_miou

from ..config.model_config import UMPAModelConfig
from ..data.polyp_dataset import DatasetConfig, PolypDataset, collate_fn
from ..umpa_model import UMPAModel

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


def benchmark_test(
    *,
    checkpoint_path: str,
    dataset_cfg: DatasetConfig,
    model_config: UMPAModelConfig,
    batch_size: int = 4,
    num_workers: int = 4,
    device: str = "cuda",
    transform=None,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    use_amp: bool = False,
    fallback_to_val: bool = False,
) -> Dict[str, Any]:
    """
    Run a benchmark on the test split and return metrics + timing.

    Parameters
    ----------
    checkpoint_path : str
        Path to a trained UMPA checkpoint (.pth).
    dataset_cfg : DatasetConfig
        Dataset configuration for PolypDataset.
    model_config : UMPAModelConfig
        Model configuration (embed dims, SAM checkpoint, etc.).
    batch_size, num_workers, device : loader/runtime settings.
    transform : optional albumentations.Compose override for test split.
    pin_memory, persistent_workers : DataLoader options.
    use_amp : bool
        Enable AMP during evaluation (speed, possibly lower precision).
    fallback_to_val : bool
        If True and test split is missing, fall back to validation split.
    """
    split = "test"
    try:
        test_dataset = PolypDataset(cfg=dataset_cfg, phase=split)
    except ValueError:
        if not fallback_to_val:
            raise
        split = "val"
        test_dataset = PolypDataset(cfg=dataset_cfg, phase=split)

    if transform is not None:
        test_dataset.transform = transform

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # Ensure checkpoint_path is reflected in the model config (without mutation).
    if getattr(model_config, "checkpoint_path", None) != checkpoint_path:
        try:
            model_config = replace(model_config, checkpoint_path=checkpoint_path)
        except TypeError:
            model_config.checkpoint_path = checkpoint_path

    map_location = "cpu" if device.startswith("cuda") else device
    model = UMPAModel.from_config(
        model_config=model_config,
        checkpoint_path=checkpoint_path,
        map_location=map_location,
    ).to(device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    start = time.perf_counter()

    amp_device = "cuda" if device.startswith("cuda") else "cpu"
    with torch.inference_mode():
        with torch.amp.autocast(amp_device, enabled=use_amp):
            metrics = evaluate(model, test_loader, device=device)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    num_samples = len(test_dataset)
    metrics.update(
        {
            "split": split,
            "num_samples": num_samples,
            "elapsed_sec": elapsed,
            "img_per_sec": (num_samples / elapsed) if elapsed > 0 else float("inf"),
            "sec_per_img": (elapsed / num_samples) if num_samples > 0 else float("inf"),
        }
    )
    return metrics
