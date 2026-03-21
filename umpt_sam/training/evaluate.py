# umpt_sam/evaluate.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sam3.train.loss.loss_fns import segment_miou

from ..config.model_config import UMPAModelConfig
from ..data.polyp_dataset import DatasetConfig, PolypDataset, collate_fn
from ..umpa_model import UMPAModel

def evaluate(model, val_loader, device="cuda", threshold=0.5, full_metrics=False):
    """
    Đánh giá mô hình.

    Parameters
    ----------
    full_metrics : bool
        - False (default): chỉ trả dice + miou (dùng cho val mỗi epoch, nhanh).
        - True: trả đầy đủ 10 metrics (dùng cho test cuối cùng):
          miou, dice, tp, fp, fn, precision, recall, f2, mask_ap, num_samples.
    """
    model.eval()

    total_miou = 0.0
    total_dice = 0.0
    num_batches = 0

    # Full metrics accumulators (only when full_metrics=True)
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
            images = batch['image'].to(device)
            gt_masks = batch['mask'].to(device)

            boxes = batch.get('bbox', None)
            points = batch.get('points', None)
            point_labels = batch.get('point_labels', None)
            captions = batch.get('text', None)

            if boxes is not None: boxes = boxes.to(device)
            if points is not None: points = points.to(device)
            if point_labels is not None: point_labels = point_labels.to(device)

            outputs = model(
                image=images,
                boxes=boxes,
                points=points,
                point_labels=point_labels,
                captions=captions,
            )
            pred_masks = outputs['pred_masks']

            if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                pred_masks = F.interpolate(
                    pred_masks, size=gt_masks.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            pred_binary = (torch.sigmoid(pred_masks) > threshold).float()

            # --- mIoU (SAM3 segment_miou) ---
            pred_3d = pred_binary.squeeze(1).bool()
            gt_3d = (gt_masks.squeeze(1) > 0.5).bool()
            batch_miou = segment_miou(pred_3d, gt_3d)

            # --- Dice ---
            intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
            union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
            dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

            total_miou += batch_miou.item()
            total_dice += dice.mean().item()
            num_batches += 1

            if full_metrics:
                # --- TP / FP / FN ---
                tp = ((pred_binary == 1.0) & (gt_masks == 1.0)).sum(dim=(1, 2, 3))
                fp = ((pred_binary == 1.0) & (gt_masks == 0.0)).sum(dim=(1, 2, 3))
                fn = ((pred_binary == 0.0) & (gt_masks == 1.0)).sum(dim=(1, 2, 3))

                # --- Precision / Recall / F2 ---
                precision = (tp.float() + 1e-6) / (tp.float() + fp.float() + 1e-6)
                recall = (tp.float() + 1e-6) / (tp.float() + fn.float() + 1e-6)
                f2 = (
                    (1 + 2**2) * precision * recall
                    / ((2**2) * precision + recall + 1e-6)
                )

                # --- Mask AP (IoU thresholds 0.50..0.95) ---
                iou = intersection.float() / (union.float() - intersection.float() + 1e-6)
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

                pbar.set_postfix({
                    "Dice": f"{dice.mean().item():.4f}",
                    "mIoU": f"{batch_miou.item():.4f}",
                    "Prec": f"{precision.mean().item():.4f}",
                    "Rec": f"{recall.mean().item():.4f}",
                    "F2": f"{f2.mean().item():.4f}",
                })
            else:
                pbar.set_postfix({
                    "Dice": f"{dice.mean().item():.4f}",
                    "mIoU": f"{batch_miou.item():.4f}",
                })

    if num_batches == 0:
        base = {"miou": 0.0, "dice": 0.0}
        if full_metrics:
            base.update({
                "tp": 0, "fp": 0, "fn": 0,
                "precision": 0.0, "recall": 0.0, "f2": 0.0,
                "mask_ap": 0.0, "num_samples": 0,
            })
        return base

    result = {
        "miou": total_miou / num_batches,
        "dice": total_dice / num_batches,
    }

    if full_metrics:
        result.update({
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_precision / num_batches,
            "recall": total_recall / num_batches,
            "f2": total_f2 / num_batches,
            "mask_ap": total_mask_ap / num_batches,
            "num_samples": total_samples,
        })

    return result


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
