"""SAM3 Baseline Evaluator with Perturbed Prompts.

Runs vanilla SAM3 (frozen, no UPFE, no MPCL) on the same dataset and
perturbation conditions as UMPA-SAM.  This produces the baseline numbers
needed for the ablation table:

    ┌──────────────────────┬──────────┬──────────┐
    │ Method               │ Dice     │ mIoU     │
    ├──────────────────────┼──────────┼──────────┤
    │ SAM3 (clean prompts) │  X.XX    │  X.XX    │
    │ SAM3 (perturbed)     │  X.XX    │  X.XX    │
    │ UMPA-SAM (perturbed) │  0.92    │  X.XX    │
    └──────────────────────┴──────────┴──────────┘

Usage:
    python eval_sam3_baseline.py --sam_checkpoint sam3.pt \
                                 --dataset_root /path/to/kvasir-sessile

The script evaluates SAM3 under multiple perturbation levels:
  1. Clean prompts (ground-truth bbox, points, text — no noise)
  2. Light perturbation  (σ_bbox=5,  σ_point=3)
  3. Medium perturbation (σ_bbox=15, σ_point=10)  — default MPPG level
  4. Heavy perturbation  (σ_bbox=30, σ_point=20)
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sam3.train.loss.loss_fns import segment_miou
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
from umpt_sam.data.kvasir_sessile import DATASET_SOURCE, TRANSFORM_PIPELINE


# ──────────────────────────────────────────────────────────────────────
# Prompt perturbation (standalone, no learnable params)
# ──────────────────────────────────────────────────────────────────────

def perturb_bbox(bbox: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to bounding box coordinates.

    bbox: (B, N, 4) or (B, 4)  —  xyxy format
    sigma: standard deviation of noise in pixels
    """
    if sigma <= 0:
        return bbox
    noise = torch.randn_like(bbox) * sigma
    perturbed = bbox + noise
    # Ensure x1 < x2, y1 < y2
    if perturbed.dim() == 3:
        perturbed[..., 0] = torch.min(perturbed[..., 0], perturbed[..., 2] - 1)
        perturbed[..., 1] = torch.min(perturbed[..., 1], perturbed[..., 3] - 1)
    elif perturbed.dim() == 2:
        perturbed[:, 0] = torch.min(perturbed[:, 0], perturbed[:, 2] - 1)
        perturbed[:, 1] = torch.min(perturbed[:, 1], perturbed[:, 3] - 1)
    return perturbed


def perturb_points(points: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise to point coordinates.

    points: (B, N, 2) — xy format
    sigma: standard deviation of noise in pixels
    """
    if sigma <= 0:
        return points
    noise = torch.randn_like(points) * sigma
    return points + noise


# ──────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────

def load_vanilla_sam3(sam_checkpoint: str, device: str = "cuda"):
    """Load SAM3 with untrained UPFE (baseline)."""
    from umpt_sam.umpa_model import UMPAModel
    from umpt_sam.config.model_config import UMPAModelConfig

    config = UMPAModelConfig(
        sam_checkpoint=sam_checkpoint,
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
    )
    model = UMPAModel.from_config(model_config=config)
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_umpa_trained(sam_checkpoint: str, umpa_checkpoint: str, device: str = "cuda"):
    """Load trained UMPA-SAM from checkpoint.

    Uses the same two-step loading as polyp_evaluator.py:
      1. Build model with SAM3 base weights (checkpoint_path=None)
      2. Overlay trained UMPA weights on top via load_weights()
    """
    from umpt_sam.umpa_model import UMPAModel
    from umpt_sam.config.model_config import UMPAModelConfig

    config = UMPAModelConfig(
        sam_checkpoint=sam_checkpoint,
        checkpoint_path=umpa_checkpoint,
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
    )
    # Step 1: Build with SAM3 base weights
    model = UMPAModel.from_config(
        model_config=config,
        checkpoint_path=None,       # load SAM3 weights first
    )
    # Step 2: Overlay trained UMPA weights
    model.load_weights(
        umpa_checkpoint,
        skip_image_encoder=False,
        map_location="cpu",
    )
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


# ──────────────────────────────────────────────────────────────────────
# Evaluation loop
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_sam3_baseline(
    model,
    dataloader: DataLoader,
    bbox_sigma: float = 0.0,
    point_sigma: float = 0.0,
    device: str = "cuda",
    label: str = "",
) -> dict:
    """Evaluate vanilla SAM3 (via UMPA model, but UPFE contributes random weights).

    To simulate *pure* SAM3, we use the model's forward() WITHOUT training,
    which means UPFE produces untrained fusion. The key comparison is:
      - SAM3 baseline: untrained UMPA model (UPFE = random init)
      - UMPA-SAM:      trained UMPA model (UPFE = learned fusion)

    Both receive the same perturbed prompts.
    """
    model.eval()
    total_dice = 0.0
    total_miou = 0.0
    n_batches = 0

    desc = f"SAM3 Baseline [{label}] σ_bbox={bbox_sigma}, σ_point={point_sigma}"
    pbar = tqdm(dataloader, desc=desc)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        gt_masks = batch["mask"].to(device, non_blocking=True)

        boxes = batch.get("bbox", None)
        points = batch.get("points", None)
        point_labels = batch.get("point_labels", None)
        captions = batch.get("text", None)

        if boxes is not None:
            boxes = boxes.to(device, non_blocking=True)
        if points is not None:
            points = points.to(device, non_blocking=True)
        if point_labels is not None:
            point_labels = point_labels.to(device, non_blocking=True)

        # ── Apply perturbation (外部, giống MPPG nhưng không qua module) ──
        if boxes is not None:
            boxes = perturb_bbox(boxes, bbox_sigma)
        if points is not None:
            points = perturb_points(points, point_sigma)

        # ── Forward qua model (UPFE chưa train = gần như identity) ──
        outputs = model(
            image=images,
            boxes=boxes,
            points=points,
            point_labels=point_labels,
            captions=captions,
        )
        pred_masks = outputs["pred_masks"]

        # Resize pred to match GT if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            pred_masks = F.interpolate(
                pred_masks, size=gt_masks.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        pred_binary = (torch.sigmoid(pred_masks) > 0.5).float()

        # ── Dice ──
        intersection = (pred_binary * gt_masks).sum(dim=(1, 2, 3))
        union = pred_binary.sum(dim=(1, 2, 3)) + gt_masks.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        mean_dice = dice.mean().item()
        total_dice += mean_dice

        # ── mIoU ──
        pred_3d = pred_binary.squeeze(1).bool()
        gt_3d = (gt_masks.squeeze(1) > 0.5).bool()
        batch_miou = segment_miou(pred_3d, gt_3d)
        total_miou += batch_miou.item()

        n_batches += 1
        pbar.set_postfix(Dice=f"{mean_dice:.4f}", mIoU=f"{batch_miou.item():.4f}")

    avg_dice = total_dice / max(n_batches, 1)
    avg_miou = total_miou / max(n_batches, 1)

    return {"dice": avg_dice, "miou": avg_miou, "label": label}


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

PERTURBATION_LEVELS = {
    "clean":  {"bbox_sigma": 0.0,  "point_sigma": 0.0},
    "light":  {"bbox_sigma": 5.0,  "point_sigma": 3.0},
    "medium": {"bbox_sigma": 15.0, "point_sigma": 10.0},
    "heavy":  {"bbox_sigma": 30.0, "point_sigma": 20.0},
}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vanilla SAM3 under perturbed prompts (ablation baseline)"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default="sam3.pt",
        help="Path to SAM3 checkpoint (.pt)",
    )
    parser.add_argument(
        "--dataset_root", type=str, default=None,
        help="Dataset root (default: use DATASET_SOURCE from kvasir_sessile.py)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--levels", type=str, nargs="+",
        default=list(PERTURBATION_LEVELS.keys()),
        choices=list(PERTURBATION_LEVELS.keys()),
        help="Perturbation levels to evaluate",
    )
    parser.add_argument(
        "--n_runs", type=int, default=3,
        help="Number of evaluation runs per level (average over randomness)",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["val", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--umpa_checkpoint", type=str, default=None,
        help="Path to trained UMPA-SAM checkpoint (.pth). If provided, also evaluates UMPA-SAM.",
    )
    args = parser.parse_args()

    # ── Dataset ──
    ds_root = args.dataset_root or DATASET_SOURCE
    dataset_config = DatasetConfig.kvasir_sessile(root=ds_root)
    dataset = PolypDataset(cfg=dataset_config, phase=args.split)
    dataset.transform = TRANSFORM_PIPELINE["val"]

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn,
    )
    print(f"Dataset: {ds_root} | Split: {args.split} | Samples: {len(dataset)}")

    # ── Model ──
    print(f"Loading vanilla SAM3 from: {args.sam_checkpoint}")
    model = load_vanilla_sam3(args.sam_checkpoint, device=args.device)
    print("SAM3 loaded (all weights frozen, UPFE untrained).\n")

    # ── Evaluate all perturbation levels ──
    results = []
    print("=" * 70)
    print(f"{'Level':<10} {'σ_bbox':>8} {'σ_point':>8} {'Dice':>10} {'mIoU':>10}")
    print("-" * 70)

    for level_name in args.levels:
        params = PERTURBATION_LEVELS[level_name]
        run_dices, run_mious = [], []

        for run_i in range(args.n_runs):
            label = f"{level_name} (run {run_i+1}/{args.n_runs})"
            metrics = evaluate_sam3_baseline(
                model, dataloader,
                bbox_sigma=params["bbox_sigma"],
                point_sigma=params["point_sigma"],
                device=args.device,
                label=label,
            )
            run_dices.append(metrics["dice"])
            run_mious.append(metrics["miou"])

        avg_dice = np.mean(run_dices)
        std_dice = np.std(run_dices)
        avg_miou = np.mean(run_mious)
        std_miou = np.std(run_mious)

        result = {
            "level": level_name,
            "bbox_sigma": params["bbox_sigma"],
            "point_sigma": params["point_sigma"],
            "dice_mean": avg_dice,
            "dice_std": std_dice,
            "miou_mean": avg_miou,
            "miou_std": std_miou,
        }
        results.append(result)

        print(
            f"{level_name:<10} {params['bbox_sigma']:>8.1f} {params['point_sigma']:>8.1f} "
            f"{avg_dice:>7.4f}±{std_dice:.4f} {avg_miou:>7.4f}±{std_miou:.4f}"
        )

    print("=" * 70)

    # ── Evaluate UMPA-SAM (trained) if checkpoint is provided ──
    umpa_results = []
    if args.umpa_checkpoint:
        print(f"\nLoading trained UMPA-SAM from: {args.umpa_checkpoint}")
        umpa_model = load_umpa_trained(
            args.sam_checkpoint, args.umpa_checkpoint, device=args.device
        )
        print("UMPA-SAM loaded (trained weights).\n")

        print("=" * 70)
        print(f"{'Level':<10} {'σ_bbox':>8} {'σ_point':>8} {'Dice':>10} {'mIoU':>10}")
        print("-" * 70)

        for level_name in args.levels:
            params = PERTURBATION_LEVELS[level_name]
            run_dices, run_mious = [], []

            for run_i in range(args.n_runs):
                label = f"UMPA {level_name} (run {run_i+1}/{args.n_runs})"
                metrics = evaluate_sam3_baseline(
                    umpa_model, dataloader,
                    bbox_sigma=params["bbox_sigma"],
                    point_sigma=params["point_sigma"],
                    device=args.device,
                    label=label,
                )
                run_dices.append(metrics["dice"])
                run_mious.append(metrics["miou"])

            avg_dice = np.mean(run_dices)
            std_dice = np.std(run_dices)
            avg_miou = np.mean(run_mious)
            std_miou = np.std(run_mious)

            umpa_result = {
                "level": level_name,
                "bbox_sigma": params["bbox_sigma"],
                "point_sigma": params["point_sigma"],
                "dice_mean": avg_dice,
                "dice_std": std_dice,
                "miou_mean": avg_miou,
                "miou_std": std_miou,
            }
            umpa_results.append(umpa_result)

            print(
                f"{level_name:<10} {params['bbox_sigma']:>8.1f} {params['point_sigma']:>8.1f} "
                f"{avg_dice:>7.4f}±{std_dice:.4f} {avg_miou:>7.4f}±{std_miou:.4f}"
            )

        print("=" * 70)
        del umpa_model
        torch.cuda.empty_cache()

    # ── Final summary table ──
    print("\n" + "=" * 80)
    print("ABLATION TABLE FOR PAPER")
    print("=" * 80)
    print(f"{'Method':<25} {'Perturbation':<15} {'Dice':>18} {'mIoU':>18}")
    print("-" * 80)
    for r in results:
        print(
            f"{'SAM3 (baseline)':<25} {r['level']:<15} "
            f"{r['dice_mean']:.4f}±{r['dice_std']:.4f}  "
            f"{r['miou_mean']:.4f}±{r['miou_std']:.4f}"
        )
    if umpa_results:
        print("-" * 80)
        for r in umpa_results:
            print(
                f"{'UMPA-SAM (ours)':<25} {r['level']:<15} "
                f"{r['dice_mean']:.4f}±{r['dice_std']:.4f}  "
                f"{r['miou_mean']:.4f}±{r['miou_std']:.4f}"
            )
    print("=" * 80)


if __name__ == "__main__":
    main()
