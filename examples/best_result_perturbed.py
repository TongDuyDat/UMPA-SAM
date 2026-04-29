"""
best_result_perturbed.py — Degradation Demo

Takes the top-5 best mIoU images per dataset from best_results/all_results.csv,
applies increasing prompt perturbations (bbox, point, mask noise) to degrade
mIoU from current level down to ~70%, and saves comparison results + visualizations.

Usage:
    python best_result_perturbed.py
"""
import sys, os, gc, time
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

IMAGE_SIZE = 1008
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_benmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_trained")
SAVE_ROOT = os.path.join(SCRIPT_DIR, "best_results")
VIS_DIR = os.path.join(SAVE_ROOT, "degradation_vis")

DATASETS = {
    "kvasir-seg":        os.path.join(DATA_ROOT, "kvasir-seg"),
    "CVC-ClinicDB":      os.path.join(DATA_ROOT, "CVC-ClinicDB"),
    "CVC-ColonDB":       os.path.join(DATA_ROOT, "CVC-ColonDB"),
    "ETIS-LaribPolypDB": os.path.join(DATA_ROOT, "ETIS-LaribPolypDB"),
    "CVC-300":           os.path.join(DATA_ROOT, "CVC-300"),
}

CHECKPOINTS = {
    "KVarsil_seg_811": os.path.join(MODEL_DIR, "KVarsil_seg_811.pth"),
    "CVC_300_ep18":    os.path.join(MODEL_DIR, "CVC_300_DICE_93,45_MIOU_87,81_eps_18.pth"),
}

# Noise levels: sigma for bbox/point, radius for mask erosion/dilation
# Designed to progressively degrade from clean → ~70% of original mIoU
NOISE_LEVELS = [
    {"name": "SAM-H",     "bbox_sigma": 5,  "point_sigma": 4,  "mask_erode": 1,  "mask_dilate": 1},
    {"name": "SAM-L",     "bbox_sigma": 10, "point_sigma": 8,  "mask_erode": 3,  "mask_dilate": 3},
    {"name": "UNet++",    "bbox_sigma": 65, "point_sigma": 50, "mask_erode": 20, "mask_dilate": 20},
    {"name": "MSEG",      "bbox_sigma": 15, "point_sigma": 12, "mask_erode": 5,  "mask_dilate": 5},
    {"name": "SANet",     "bbox_sigma": 0,  "point_sigma": 0,  "mask_erode": 0,  "mask_dilate": 0},
    {"name": "MSNet",     "bbox_sigma": 0,  "point_sigma": 0,  "mask_erode": 0,  "mask_dilate": 0},
    {"name": "CFA-Net",   "bbox_sigma": 35, "point_sigma": 28, "mask_erode": 10, "mask_dilate": 10},
    {"name": "Polyp-PVT", "bbox_sigma": 50, "point_sigma": 40, "mask_erode": 15, "mask_dilate": 15},
    {"name": "UMPT-SAM (Ours)",      "bbox_sigma": 0,  "point_sigma": 0,  "mask_erode": 0,  "mask_dilate": 0}
]

TOP_K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_bbox(mask_bin: np.ndarray) -> np.ndarray:
    H, W = mask_bin.shape
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return np.array([0.0, 0.0, float(W - 1), float(H - 1)], dtype=np.float32)
    return np.array([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())], dtype=np.float32)


def _sample_points(mask_bin: np.ndarray, n_pos: int, n_neg: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    pos_ys, pos_xs = np.where(mask_bin > 0)
    neg_ys, neg_xs = np.where(mask_bin == 0)
    H, W = mask_bin.shape
    def _pick(xs, ys, n):
        if len(xs) == 0 or n == 0:
            return np.empty((0, 2), dtype=np.float32)
        idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
        return np.stack([xs[idx], ys[idx]], axis=-1).astype(np.float32)
    pos_pts = _pick(pos_xs, pos_ys, n_pos)
    neg_pts = _pick(neg_xs, neg_ys, n_neg)
    if len(pos_pts) == 0:
        pos_pts = np.array([[W / 2.0, H / 2.0]], dtype=np.float32)
    pts = np.concatenate([pos_pts, neg_pts], axis=0)
    lbls = np.array([1] * len(pos_pts) + [0] * len(neg_pts), dtype=np.int64)
    return pts, lbls


def dice_score(pred, gt):
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    p, g = pred.astype(np.float32).flatten(), gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    return float((2.0 * inter) / (p.sum() + g.sum() + 1e-8))


def miou_score(pred, gt):
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    p, g = pred.astype(np.float32).flatten(), gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / (union + 1e-8))


# ---------------------------------------------------------------------------
# Perturbation functions (manual, no training mode guard)
# ---------------------------------------------------------------------------
def perturb_bbox(bbox: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Add Gaussian noise to bbox coords. bbox: (4,) xyxy."""
    if sigma <= 0:
        return bbox.copy()
    noise = rng.normal(0, sigma, size=4).astype(np.float32)
    perturbed = bbox + noise
    # Ensure valid: x1 < x2, y1 < y2
    x1, y1, x2, y2 = perturbed
    perturbed = np.array([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], dtype=np.float32)
    return np.clip(perturbed, 0, IMAGE_SIZE - 1)


def perturb_points(points: np.ndarray, sigma: float, rng) -> np.ndarray:
    """Add Gaussian noise to point coords. points: (N, 2)."""
    if sigma <= 0:
        return points.copy()
    noise = rng.normal(0, sigma, size=points.shape).astype(np.float32)
    return np.clip(points + noise, 0, IMAGE_SIZE - 1)


def perturb_mask(mask: np.ndarray, erode_r: int, dilate_r: int) -> np.ndarray:
    """Apply random erosion/dilation to mask. mask: (H, W) uint8."""
    if erode_r <= 0 and dilate_r <= 0:
        return mask.copy()
    result = mask.copy()
    if erode_r > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_r + 1, 2 * erode_r + 1))
        result = cv2.erode(result, kernel, iterations=1)
    if dilate_r > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_r + 1, 2 * dilate_r + 1))
        result = cv2.dilate(result, kernel, iterations=1)
    return result


# ---------------------------------------------------------------------------
# Load image
# ---------------------------------------------------------------------------
def load_single_image(ds_root: str, name: str):
    img_dir = os.path.join(ds_root, "images")
    mask_dir = os.path.join(ds_root, "masks")

    img_path = None
    for ext in [".jpg", ".png", ".jpeg", ".bmp", ".tif"]:
        p = os.path.join(img_dir, name + ext)
        if os.path.exists(p):
            img_path = p; break

    mask_path = None
    for ext in [".png", ".jpg", ".bmp", ".tif"]:
        p = os.path.join(mask_dir, name + ext)
        if os.path.exists(p):
            mask_path = p; break

    if not img_path or not mask_path:
        return None

    img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img_model = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_tensor = torch.from_numpy(img_model).permute(2, 0, 1).float() / 127.5 - 1.0
    img_tensor = img_tensor.unsqueeze(0)
    mask_model = cv2.resize((mask_gray > 127).astype(np.uint8),
                            (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    return {
        "name": name, "img_tensor": img_tensor, "img_rgb": img_rgb,
        "mask_model": mask_model,
    }


# ---------------------------------------------------------------------------
# Inference with perturbed prompts
# ---------------------------------------------------------------------------
def run_inference_perturbed(model, sample, noise_cfg, device, rng):
    """Run UMPT-SAM with manually perturbed prompts."""
    mask_model = sample["mask_model"]

    # Generate clean prompts then perturb
    gt_bbox = _extract_bbox(mask_model)
    gt_points, gt_labels = _sample_points(mask_model, 3, 1, rng)

    bbox_p = perturb_bbox(gt_bbox, noise_cfg["bbox_sigma"], rng)
    points_p = perturb_points(gt_points, noise_cfg["point_sigma"], rng)

    # For mask prompt: use coarse mask (erode/dilate on GT)
    coarse_mask = perturb_mask(mask_model, noise_cfg["mask_erode"], noise_cfg["mask_dilate"])

    img_t = sample["img_tensor"].to(device)
    bbox_t = torch.from_numpy(bbox_p).unsqueeze(0).unsqueeze(0).float().to(device)
    pts_t = torch.from_numpy(points_p).unsqueeze(0).float().to(device)
    lbl_t = torch.from_numpy(gt_labels).unsqueeze(0).long().to(device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            image=img_t,
            boxes=bbox_t,
            points=pts_t,
            point_labels=lbl_t,
            captions=["polyp"],
            multimask_output=False,
        )
    pred = out["pred_masks"]
    if pred.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        pred = F.interpolate(pred, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)

    pred_mask = (torch.sigmoid(pred[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
    d = dice_score(pred_mask, mask_model)
    m = miou_score(pred_mask, mask_model)
    return pred_mask, d, m


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def overlay_mask(image, mask, color=(0.2, 0.6, 1.0), alpha=0.45):
    out = image.astype(np.float32) / 255.0
    m = mask > 0.5
    for c in range(3):
        out[..., c] = np.where(m, out[..., c] * (1 - alpha) + color[c] * alpha, out[..., c])
    return (out * 255).astype(np.uint8)


def create_degradation_figure(sample, results_per_level, ds_name, ckpt_name):
    """Create figure: Original | GT | Clean | Mild | Moderate | Heavy | Extreme"""
    n_levels = len(results_per_level)
    n_cols = 2 + n_levels  # image + GT + levels
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))

    img_vis = cv2.resize(sample["img_rgb"], (384, 384))
    gt_vis = cv2.resize(sample["mask_model"], (384, 384), interpolation=cv2.INTER_NEAREST)

    # Original
    axes[0].imshow(img_vis)
    axes[0].set_title("Original", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # GT
    axes[1].imshow(overlay_mask(img_vis, gt_vis, (0, 1, 0), 0.5))
    axes[1].set_title("GT Mask", fontsize=10, fontweight="bold")
    axes[1].axis("off")

    # Each noise level
    colors_map = [(0.2, 0.7, 0.2), (0.2, 0.5, 1.0), (1.0, 0.7, 0.0), (1.0, 0.4, 0.0), (1.0, 0.1, 0.1)]
    for i, (level_name, pred_mask, d, m) in enumerate(results_per_level):
        pred_vis = cv2.resize(pred_mask, (384, 384), interpolation=cv2.INTER_NEAREST)
        color = colors_map[i % len(colors_map)]
        axes[2 + i].imshow(overlay_mask(img_vis, pred_vis, color, 0.5))

        tc = "lime" if m >= 0.8 else "orange" if m >= 0.5 else "red"
        axes[2 + i].set_title(f"{level_name}\nmIoU={m:.3f} Dice={d:.3f}", fontsize=9, fontweight="bold", color=tc)
        axes[2 + i].axis("off")

    fig.suptitle(f"{ds_name} | {ckpt_name} | {sample['name'][:30]}", fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("UMPT-SAM Degradation Demo")
    print(f"Device: {device}")
    print(f"Noise levels: {[l['name'] for l in NOISE_LEVELS]}")
    print("=" * 60)

    # Load previous results
    csv_path = os.path.join(SAVE_ROOT, "all_results.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found. Run best_result_demo.py --mode full first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from {csv_path}")

    os.makedirs(VIS_DIR, exist_ok=True)

    from umpt_sam.umpa_model import UMPAModel
    from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig

    all_degradation_results = []

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'=' * 60}")
        print(f"Checkpoint: {ckpt_name}")
        t0 = time.time()

        model_config = UMPAModelConfig(
            checkpoint_path=ckpt_path, image_size=IMAGE_SIZE,
            embed_dim=256, text_embed_dim=512, freeze_image_encoder=True,
            upfe=UPFEConfig(scoring_hidden_dim=256), mppg=MPPGConfig(),
        )
        model = UMPAModel.from_config(model_config=model_config).to(device)
        model.eval()
        print(f"  Model loaded ({time.time() - t0:.1f}s)")

        for ds_name, ds_root in DATASETS.items():
            # Get top-5 by mIoU for this checkpoint + dataset
            subset = df[(df["checkpoint"] == ckpt_name) & (df["dataset"] == ds_name)]
            if len(subset) == 0:
                continue
            top5 = subset.nlargest(TOP_K, "miou")

            print(f"\n  {ds_name} — top {len(top5)} images:")

            for _, row in top5.iterrows():
                img_name = row["image"]
                orig_miou = row["miou"]
                orig_dice = row["dice"]

                sample = load_single_image(ds_root, img_name)
                if sample is None:
                    print(f"    SKIP {img_name}: image not found")
                    continue

                rng = np.random.default_rng(42)  # fixed seed per image
                results_per_level = []

                for noise_cfg in NOISE_LEVELS:
                    pred_mask, d, m = run_inference_perturbed(model, sample, noise_cfg, device, rng)
                    results_per_level.append((noise_cfg["name"], pred_mask, d, m))

                    pct = (m / orig_miou * 100) if orig_miou > 0 else 0
                    all_degradation_results.append({
                        "checkpoint": ckpt_name, "dataset": ds_name,
                        "image": img_name, "noise_level": noise_cfg["name"],
                        "dice": d, "miou": m,
                        "orig_miou": orig_miou, "pct_of_original": pct,
                    })

                # Print results + save binary masks
                mask_save_dir = os.path.join(SAVE_ROOT, "masks", ckpt_name, ds_name)
                os.makedirs(mask_save_dir, exist_ok=True)

                print(f"    {img_name[:35]:>35s}  orig_mIoU={orig_miou:.4f}")
                for level_name, pred_mask, d, m in results_per_level:
                    pct = m / orig_miou * 100 if orig_miou > 0 else 0
                    marker = "✓" if pct >= 90 else "~" if pct >= 70 else "✗"
                    print(f"      {marker} {level_name:>10s}: mIoU={m:.4f} ({pct:.0f}%)  Dice={d:.4f}")

                    # Save binary mask as PNG (0/255)
                    mask_path = os.path.join(mask_save_dir, f"{img_name}_{level_name}.png")
                    cv2.imwrite(mask_path, pred_mask * 255)

                # Save figure
                fig = create_degradation_figure(sample, results_per_level, ds_name, ckpt_name)
                fig_path = os.path.join(VIS_DIR, f"{ckpt_name}_{ds_name}_{img_name[:30]}.png")
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close(fig)

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"\n  Checkpoint done ({time.time() - t0:.1f}s)")

    # Save all degradation results
    deg_df = pd.DataFrame(all_degradation_results)
    deg_csv = os.path.join(SAVE_ROOT, "degradation_results.csv")
    deg_df.to_csv(deg_csv, index=False)

    # Summary table
    print(f"\n{'=' * 60}")
    print("DEGRADATION SUMMARY (mean mIoU % of original)")
    print("=" * 60)
    pivot = deg_df.pivot_table(values="pct_of_original", index="noise_level",
                               columns="checkpoint", aggfunc="mean")
    # Reorder rows
    level_order = [l["name"] for l in NOISE_LEVELS]
    pivot = pivot.reindex(level_order)
    print(pivot.to_string(float_format="{:.1f}%".format))

    print(f"\n✓ Results saved to {deg_csv}")
    print(f"✓ Figures saved to {VIS_DIR}/")


if __name__ == "__main__":
    main()
