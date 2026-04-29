"""Evaluation utilities for UMPAv2 models.

Adapted from ``umpt_sam.training.evaluate`` with UMPAv2 forward signature.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from sam3.train.loss.loss_fns import segment_miou


def evaluate(
    model,
    val_loader,
    device: str = "cuda",
    threshold: float = 0.5,
    full_metrics: bool = False,
) -> Dict[str, Any]:
    """Evaluate an UMPAv2 model.

    Parameters
    ----------
    model : UMPAv2Model
        Model to evaluate (or wrapper with compatible forward).
    val_loader : DataLoader
        Validation / test loader.
    device : str
        Device string.
    threshold : float
        Binarisation threshold for predicted masks.
    full_metrics : bool
        If True, return 10 metrics (dice, miou, precision, recall, f2,
        mask_ap, tp, fp, fn, num_samples).  Otherwise just dice + miou.
    """
    model.eval()

    total_miou = 0.0
    total_dice = 0.0
    num_batches = 0

    if full_metrics:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_precision = 0.0
        total_recall = 0.0
        total_f2 = 0.0
        total_mask_ap = 0.0
        total_samples = 0

    desc = "Testing (full metrics)" if full_metrics else "Validating"
    pbar = tqdm(val_loader, desc=desc)

    with torch.inference_mode():
        for batch in pbar:
            images = batch["image"].to(device)
            gt_masks = batch["mask"].to(device)

            boxes = batch.get("bbox", None)
            points = batch.get("points", None)
            point_labels = batch.get("point_labels", None)
            captions = batch.get("text", None)
            masks_input = batch.get("coarse_mask", None)

            if boxes is not None:
                boxes = boxes.to(device)
            if points is not None:
                points = points.to(device)
            if point_labels is not None:
                point_labels = point_labels.to(device)
            if masks_input is not None:
                masks_input = masks_input.to(device)

            # UMPAv2 forward signature
            outputs = model(
                image=images,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                masks=masks_input,
                captions=captions,
            )
            pred_masks = outputs["pred_masks"]

            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks,
                    size=gt_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

            # mIoU
            pred_3d = pred_binary.squeeze(1).bool()
            gt_3d = (gt_masks.squeeze(1) > 0.5).bool()
            batch_miou = segment_miou(pred_3d, gt_3d)

            # Dice
            intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
            union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

            total_miou += batch_miou.item()
            total_dice += dice.mean().item()
            num_batches += 1

            if full_metrics:
                tp = ((pred_binary == 1.0) & (gt_masks == 1.0)).sum(dim=(1, 2, 3))
                fp = ((pred_binary == 1.0) & (gt_masks == 0.0)).sum(dim=(1, 2, 3))
                fn = ((pred_binary == 0.0) & (gt_masks == 1.0)).sum(dim=(1, 2, 3))

                precision = (tp.float() + 1e-6) / (tp.float() + fp.float() + 1e-6)
                recall = (tp.float() + 1e-6) / (tp.float() + fn.float() + 1e-6)
                f2 = (
                    (1 + 4) * precision * recall
                    / (4 * precision + recall + 1e-6)
                )

                iou = intersection.float() / (
                    union.float() - intersection.float() + 1e-6
                )
                thresholds = torch.arange(0.50, 0.96, 0.05, device=iou.device)
                ap_per_image = (iou.unsqueeze(1) >= thresholds).float().mean(dim=1)

                batch_size = images.shape[0]
                total_tp += int(tp.sum().item())
                total_fp += int(fp.sum().item())
                total_fn += int(fn.sum().item())
                total_precision += float(precision.mean().item())
                total_recall += float(recall.mean().item())
                total_f2 += float(f2.mean().item())
                total_mask_ap += float(ap_per_image.mean().item())
                total_samples += batch_size

                pbar.set_postfix(
                    Dice=f"{dice.mean().item():.4f}",
                    mIoU=f"{batch_miou.item():.4f}",
                    Prec=f"{precision.mean().item():.4f}",
                )
            else:
                pbar.set_postfix(
                    Dice=f"{dice.mean().item():.4f}",
                    mIoU=f"{batch_miou.item():.4f}",
                )

    if num_batches == 0:
        base: Dict[str, Any] = {"miou": 0.0, "dice": 0.0}
        if full_metrics:
            base.update(
                tp=0, fp=0, fn=0, precision=0.0, recall=0.0,
                f2=0.0, mask_ap=0.0, num_samples=0,
            )
        return base

    result: Dict[str, Any] = {
        "miou": total_miou / num_batches,
        "dice": total_dice / num_batches,
    }
    if full_metrics:
        result.update(
            tp=total_tp,
            fp=total_fp,
            fn=total_fn,
            precision=total_precision / num_batches,
            recall=total_recall / num_batches,
            f2=total_f2 / num_batches,
            mask_ap=total_mask_ap / num_batches,
            num_samples=total_samples,
        )
    return result
