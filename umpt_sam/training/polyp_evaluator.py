"""Evaluator for PolypDataset benchmarks (UMPA-SAM).

This module adds a class-based evaluator for repeatable benchmarking, without modifying
the existing evaluate.py training script.
"""

import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from sam3.train.loss.loss_fns import segment_miou

from ..config.model_config import UMPAModelConfig
from ..data.polyp_dataset import DatasetConfig, PolypDataset, collate_fn
from ..umpa_model import UMPAModel


class PolypBenchmarkEvaluator:
    """Evaluator class for benchmarking UMPA-SAM on PolypDataset."""

    def __init__(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        model_config: UMPAModelConfig,
        dataset_cfg: DatasetConfig,
        split: str = "test",
        batch_size: int = 4,
        num_workers: int = 2,
        device: str = "cuda",
        pin_memory: bool = True,
        persistent_workers: Optional[bool] = None,
        fallback_to_val: bool = True,
        **dataset_kwargs,
    ) -> "PolypBenchmarkEvaluator":
        """Load model from checkpoint and build benchmark dataloader."""
        try:
            dataset = PolypDataset(cfg=dataset_cfg, phase=split)
        except ValueError:
            if fallback_to_val and split != "val":
                split = "val"
                dataset = PolypDataset(cfg=dataset_cfg, phase=split)
            else:
                raise

        for key, value in dataset_kwargs.items():
            setattr(dataset, key, value)

        if persistent_workers is None:
            persistent_workers = num_workers > 0

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        if getattr(model_config, "checkpoint_path", None) != checkpoint_path:
            try:
                model_config = replace(model_config, checkpoint_path=checkpoint_path)
            except TypeError:
                model_config.checkpoint_path = checkpoint_path

        model = UMPAModel.from_config(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            map_location="cpu",
        ).to(device)

        return cls(model=model, dataloader=loader, device=device)

    def evaluate(self) -> Dict[str, float]:
        """Run evaluation once and return metrics."""
        self.model.eval()
        total_miou = 0.0
        total_dice = 0.0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_precision = 0.0
        total_recall = 0.0
        total_f2 = 0.0
        total_mask_ap = 0.0
        total_samples = 0

        with torch.inference_mode():
            pbar = tqdm(self.dataloader, desc="Polyp benchmark")
            for batch in pbar:
                images = batch["image"].to(self.device)
                gt_masks = batch["mask"].to(self.device)

                boxes = batch.get("bbox", None)
                points = batch.get("points", None)
                point_labels = batch.get("point_labels", None)
                text = batch.get("text", None)

                if boxes is not None:
                    boxes = boxes.to(self.device)
                if points is not None:
                    points = points.to(self.device)
                if point_labels is not None:
                    point_labels = point_labels.to(self.device)

                outputs = self.model(
                    image=images,
                    boxes=boxes,
                    points=points,
                    point_labels=point_labels,
                    captions=text,
                )
                pred_masks = outputs["pred_masks"]
                if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
                    pred_masks = F.interpolate(
                        pred_masks,
                        size=gt_masks.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                pred_binary = (torch.sigmoid(pred_masks) > self.threshold).float()
                pred_3d = pred_binary.squeeze(1).bool()
                gt_3d = (gt_masks.squeeze(1) > 0.5).bool()

                batch_miou = segment_miou(pred_3d, gt_3d)
                intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
                union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
                dice = (2.0 * intersection + 1e-6) / (union + 1e-6)

                tp = ((pred_binary == 1.0) & (gt_masks == 1.0)).sum(dim=(1, 2, 3))
                fp = ((pred_binary == 1.0) & (gt_masks == 0.0)).sum(dim=(1, 2, 3))
                fn = ((pred_binary == 0.0) & (gt_masks == 1.0)).sum(dim=(1, 2, 3))
                precision = (tp.float() + 1e-6) / (tp.float() + fp.float() + 1e-6)
                recall = (tp.float() + 1e-6) / (tp.float() + fn.float() + 1e-6)
                f2 = (
                    (1 + 2**2)
                    * precision
                    * recall
                    / ((2**2) * precision + recall + 1e-6)
                )

                # A basic mask AP as average of IoU matches at thresholds 0.5..0.95
                iou = intersection.float() / (
                    union.float() - intersection.float() + 1e-6
                )
                thresholds = torch.arange(0.50, 0.96, 0.05, device=iou.device)
                ap_per_image = (iou.unsqueeze(1) >= thresholds).float().mean(dim=1)

                batch_size = images.shape[0]
                total_miou += batch_miou.item()
                total_dice += dice.mean().item()
                total_tp += int(tp.sum().item())
                total_fp += int(fp.sum().item())
                total_fn += int(fn.sum().item())
                total_precision += float(precision.mean().item())
                total_recall += float(recall.mean().item())
                total_f2 += float(f2.mean().item())
                total_mask_ap += float(ap_per_image.mean().item())
                total_samples += batch_size

                pbar.set_postfix(
                    {
                        "mIoU": f"{batch_miou.item():.4f}",
                        "Dice": f"{dice.mean().item():.4f}",
                        "Prec": f"{precision.mean().item():.4f}",
                        "Rec": f"{recall.mean().item():.4f}",
                        "F2": f"{f2.mean().item():.4f}",
                    }
                )

        if total_samples == 0:
            return {
                "miou": 0.0,
                "dice": 0.0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f2": 0.0,
                "mask_ap": 0.0,
                "num_samples": 0,
            }

        return {
            "miou": total_miou / total_samples,
            "dice": total_dice / total_samples,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_precision / total_samples,
            "recall": total_recall / total_samples,
            "f2": total_f2 / total_samples,
            "mask_ap": total_mask_ap / total_samples,
            "num_samples": total_samples,
        }

    def benchmark(self, use_amp: bool = False) -> Dict[str, Any]:
        """Run evaluation and return metrics including timing."""
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()

        with torch.inference_mode():
            if use_amp and self.device.startswith("cuda"):
                amp_device = "cuda"
            else:
                amp_device = self.device
            with torch.amp.autocast(amp_device, enabled=use_amp):
                metrics = self.evaluate()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        num_samples = len(self.dataloader.dataset)
        metrics.update(
            {
                "elapsed_sec": elapsed,
                "img_per_sec": num_samples / elapsed if elapsed > 0 else float("inf"),
                "sec_per_img": elapsed / max(num_samples, 1),
            }
        )
        return metrics


def benchmark_from_checkpoint(
    checkpoint_path: str,
    dataset_cfg: DatasetConfig,
    model_config: UMPAModelConfig,
    split: str = "test",
    batch_size: int = 4,
    num_workers: int = 2,
    device: str = "cuda",
    use_amp: bool = False,
    fallback_to_val: bool = True,
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
) -> Dict[str, Any]:
    """Convenience API for benchmark from checkpoint."""
    evaluator = PolypBenchmarkEvaluator.from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_config=model_config,
        dataset_cfg=dataset_cfg,
        split=split,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        fallback_to_val=fallback_to_val,
    )
    return evaluator.benchmark(use_amp=use_amp)


if __name__ == "__main__":
    import argparse
    from .evaluate import benchmark_test

    parser = argparse.ArgumentParser(
        description="Run UMPA-SAM Polyp benchmark with class-based evaluator."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to UMPA checkpoint.")
    parser.add_argument(
        "--dataset-root", required=True, help="Polyp dataset root path."
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--sam-checkpoint", default="sam3.pt")
    args = parser.parse_args()

    dataset_cfg = DatasetConfig(root=args.dataset_root)
    model_config = UMPAModelConfig(
        sam_checkpoint=args.sam_checkpoint,
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
    )

    metrics = benchmark_from_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_cfg=dataset_cfg,
        model_config=model_config,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        use_amp=args.use_amp,
    )
    print("\nBenchmark summary:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
