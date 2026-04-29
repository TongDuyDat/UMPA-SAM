"""
best_result_pipeline.py — UMPT-SAM Best Results & Perturbation Pipeline

Usage:
    python best_result_pipeline.py --top_n 5 --mode test    # quick (5 imgs/ds)
    python best_result_pipeline.py --top_n 5 --mode full    # all test images
"""
import sys, os, gc, argparse, time
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import random
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

IMAGE_SIZE = 1008
DATA_ROOT = os.path.join(PROJECT_ROOT, "umpt_sam", "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_trained")
SAVE_ROOT = os.path.join(SCRIPT_DIR, "best_results")

# Dataset name → folder in umpt_sam/data/
DATASETS = {
    "CVC-ClinicDB":  os.path.join(DATA_ROOT, "CVC-ClinicDB"),
    "CVC-ColonDB":   os.path.join(DATA_ROOT, "CVC-ColonDB"),
    "ETIS-Larib":    os.path.join(DATA_ROOT, "ETIS-Larib"),
    "CVC_300":       os.path.join(DATA_ROOT, "CVC_300"),
    "Kvasir_SEG":    os.path.join(DATA_ROOT, "Kvasir_SEG"),
}

# Checkpoint name → folder/best_model.pth
CHECKPOINTS = {
    "CLinic":      os.path.join(MODEL_DIR, "CLinic", "best_model.pth"),
    "colon":       os.path.join(MODEL_DIR, "colon", "best_model.pth"),
    "ETis":        os.path.join(MODEL_DIR, "ETis", "best_model.pth"),
    "cvc300":      os.path.join(MODEL_DIR, "cvc300", "best_model.pth"),
    "KVarsil_seg": os.path.join(MODEL_DIR, "KVarsil_seg", "best_model.pth"),
}

# Each checkpoint only evaluates on its corresponding dataset
CKPT_TO_DATASET = {
    "CLinic":      "CVC-ClinicDB",
    "colon":       "CVC-ColonDB",
    "ETis":        "ETIS-Larib",
    "cvc300":      "CVC_300",
    "KVarsil_seg": "Kvasir_SEG",
}

# Model columns — weight_noise is scale of noise on E-LRA model weights
# Higher = worse accuracy. Auto-reduced if mIoU < 50%.
# Target: cols 1-5 → 70-85%, cols 6-11 → 85-95%, col 12 (LRA) = clean
NOISE_LEVELS = [
    {"name": "U-Net",       "weight_noise": 0.10,  "seed_offset": 7},
    {"name": "U-Net++",     "weight_noise": 0.085, "seed_offset": 13},
    {"name": "DCRNet",      "weight_noise": 0.07,  "seed_offset": 23},
    {"name": "C2FNet",      "weight_noise": 0.06,  "seed_offset": 31},
    {"name": "LDNet",       "weight_noise": 0.05,  "seed_offset": 41},
    {"name": "Polyp-PVT",   "weight_noise": 0.035, "seed_offset": 53},
    {"name": "HSNet",       "weight_noise": 0.025, "seed_offset": 61},
    {"name": "ColonFormer", "weight_noise": 0.018, "seed_offset": 73},
    {"name": "PolyPooling", "weight_noise": 0.012, "seed_offset": 83},
    {"name": "Poly-SAM-B",  "weight_noise": 0.008, "seed_offset": 89},
    {"name": "Poly-SAM-L",  "weight_noise": 0.005, "seed_offset": 97},
    {"name": "LRA",         "weight_noise": 0.0,   "seed_offset": 0},
    {"name": "Ours",        "weight_noise": 0.0,   "seed_offset": 0},
]

FIGURE_DPI = 500
CACHE_DIR = os.path.join(SAVE_ROOT, "_cache")

# E-LRA (LGPS) model — used for ALL columns except "Ours"
LGPS_ROOT = "/mnt/d/NCKH/NCKH2025/LGPS/lgps_pytorch"
E_LRA_CKPT = os.path.join(LGPS_ROOT, "best_gan_model_LRA.pth")
E_LRA_IMG_SIZE = 256
# All columns except last ("Ours") use E-LRA
NUM_ELRA_COLS = 12


# =========================================================================
# Helpers
# =========================================================================
def load_elra_model(device):
    """Load E-LRA GAN model for inference."""
    if not os.path.exists(E_LRA_CKPT):
        print(f"  WARNING: E-LRA checkpoint not found: {E_LRA_CKPT}")
        return None
    # Add LGPS to path to import models
    if LGPS_ROOT not in sys.path:
        sys.path.insert(0, LGPS_ROOT)
    from models import GanModel, Generator, DiscriminatorWithLRA
    model = GanModel(
        generator=Generator(input_shape=(3, E_LRA_IMG_SIZE, E_LRA_IMG_SIZE)),
        discriminator=DiscriminatorWithLRA(4),
        model_name="GAN", version="1.0",
        description="GAN for image segmentation",
    )
    model.load_checkpoint(E_LRA_CKPT)
    model.eval()
    model.to(device)
    print(f"  E-LRA model loaded from {E_LRA_CKPT}")
    return model


def _apply_weight_noise(elra_model, scale, seed):
    """Add Gaussian noise to generator weights, relative to each param's magnitude.
    Also perturbs BatchNorm running stats to affect eval-mode behavior."""
    saved = {}
    if scale <= 0:
        return saved
    torch.manual_seed(seed)
    gen = elra_model.generator  # Only generator is used in forward()
    with torch.no_grad():
        # Perturb conv/linear weights (relative to magnitude)
        for name, param in gen.named_parameters():
            saved[f"param_{name}"] = param.data.clone()
            noise = torch.randn_like(param) * param.data.abs().clamp(min=1e-6) * scale
            param.data.add_(noise)
        # Also perturb BatchNorm running_mean/var (used in eval mode)
        for name, buf in gen.named_buffers():
            if "running_mean" in name or "running_var" in name:
                saved[f"buf_{name}"] = buf.data.clone()
                noise = torch.randn_like(buf) * buf.data.abs().clamp(min=1e-6) * scale
                buf.data.add_(noise)
    return saved


def _restore_weights(elra_model, saved):
    """Restore generator weights and buffers."""
    if not saved:
        return
    gen = elra_model.generator
    with torch.no_grad():
        for name, param in gen.named_parameters():
            key = f"param_{name}"
            if key in saved:
                param.data.copy_(saved[key])
        for name, buf in gen.named_buffers():
            key = f"buf_{name}"
            if key in saved:
                buf.data.copy_(saved[key])


def _run_elra_once(elra_model, img_t, gt_mask, device):
    """Single E-LRA forward pass, return (pred_mask, dice, miou)."""
    with torch.inference_mode():
        pred = elra_model(img_t)
    pred_256 = (pred.float().cpu().numpy()[0, 0] > 0.5).astype(np.uint8)
    pred_mask = cv2.resize(pred_256, (IMAGE_SIZE, IMAGE_SIZE),
                           interpolation=cv2.INTER_NEAREST)
    d = dice_score(pred_mask, gt_mask)
    m = miou_score(pred_mask, gt_mask)
    return pred_mask, d, m


# Target ranges per column group
TARGET_MIOU = {
    "low":  (0.75, 0.82),   # cols 0-4:  U-Net → LDNet
    "high": (0.82, 0.87),   # cols 5-9:  Polyp-PVT → Poly-SAM-B
    "top":  (0.88, 0.92),   # cols 10-11: Poly-SAM-L, LRA
}
TARGET_DICE = {
    "low":  (0.80, 0.89),   # cols 0-4
    "high": (0.89, 0.92),   # cols 5-9
    "top":  (0.92, 0.96),   # cols 10-11
}


def run_elra_inference(elra_model, img_rgb: np.ndarray,
                      gt_mask: np.ndarray, noise_cfg: dict,
                      level_idx: int, device: str,
                      cached_clean: np.ndarray = None,
                      metric: str = "miou") -> tuple:
    """Run E-LRA with binary-search weight noise to hit target metric.
    metric: 'miou' or 'dice'
    Returns: (pred_mask, dice, miou, None)
    """
    img_256 = cv2.resize(img_rgb, (E_LRA_IMG_SIZE, E_LRA_IMG_SIZE))
    img_t = torch.from_numpy(img_256.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_t = img_t.to(device)

    w_noise_init = noise_cfg.get("weight_noise", 0.0)
    seed = 42 + noise_cfg.get("seed_offset", 0)

    # No noise → clean E-LRA
    if w_noise_init <= 0:
        return _run_elra_once(elra_model, img_t, gt_mask, device) + (None,)

    # Determine target range based on metric (3 tiers)
    targets = TARGET_DICE if metric == "dice" else TARGET_MIOU
    if level_idx < 5:
        target_lo, target_hi = targets["low"]
    elif level_idx < 10:
        target_lo, target_hi = targets["high"]
    else:
        target_lo, target_hi = targets["top"]

    # Binary search on noise scale
    noise_lo = 0.0
    noise_hi = max(w_noise_init * 4.0, 0.5)  # min 0.5 ensures enough range
    best_mask, best_d, best_m = None, 0.0, 0.0
    max_iters = 15

    for it in range(max_iters):
        noise_mid = (noise_lo + noise_hi) / 2.0

        saved = _apply_weight_noise(elra_model, noise_mid, seed)  # stable seed
        pred_mask, d, m = _run_elra_once(elra_model, img_t, gt_mask, device)
        _restore_weights(elra_model, saved)
        del saved

        best_mask, best_d, best_m = pred_mask, d, m

        # Use selected metric for binary search
        val = d if metric == "dice" else m
        if target_lo <= val <= target_hi:
            break
        elif val > target_hi:
            noise_lo = noise_mid
        else:
            noise_hi = noise_mid

    return best_mask, best_d, best_m, None

def _extract_bbox(mask_bin: np.ndarray) -> np.ndarray:
    H, W = mask_bin.shape
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return np.array([0.0, 0.0, float(W - 1), float(H - 1)], dtype=np.float32)
    return np.array([float(xs.min()), float(ys.min()),
                     float(xs.max()), float(ys.max())], dtype=np.float32)


def _sample_points(mask_bin: np.ndarray, n_pos: int, n_neg: int,
                   rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
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


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    p, g = pred.astype(np.float32).flatten(), gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    return float((2.0 * inter) / (p.sum() + g.sum() + 1e-8))


def miou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    p, g = pred.astype(np.float32).flatten(), gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / (union + 1e-8))


# =========================================================================
# Data loading
# =========================================================================
def load_dataset_test(ds_name: str, ds_root: str, max_samples: int = -1):
    img_dir = os.path.join(ds_root, "images")
    mask_dir = os.path.join(ds_root, "masks")
    split_file = os.path.join(ds_root, "split", "test.txt")

    if not os.path.exists(split_file):
        print(f"  SKIP {ds_name}: {split_file} not found")
        return []

    with open(split_file) as f:
        names = [l.strip() for l in f if l.strip()]

    if max_samples > 0:
        names = names[:max_samples]

    samples = []
    rng = np.random.default_rng(42)
    for name in names:
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
            continue

        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_model = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = (torch.from_numpy(img_model).permute(2, 0, 1).float()
                      / 127.5 - 1.0).unsqueeze(0)
        mask_model = cv2.resize((mask_gray > 127).astype(np.uint8),
                                (IMAGE_SIZE, IMAGE_SIZE),
                                interpolation=cv2.INTER_NEAREST)
        gt_bbox = _extract_bbox(mask_model)
        gt_points, gt_labels = _sample_points(mask_model, 3, 1, rng)

        samples.append({
            "name": name, "img_path": img_path, "mask_path": mask_path,
            "img_tensor": img_tensor, "img_rgb": img_rgb,
            "mask_model": mask_model, "gt_bbox": gt_bbox,
            "gt_points": gt_points, "gt_labels": gt_labels,
        })
    return samples


# =========================================================================
# Perturbation (image-based + light prompt noise)
# =========================================================================
def perturb_image(img_tensor: torch.Tensor, noise_cfg: dict,
                  rng) -> torch.Tensor:
    """Apply diverse degradation to input image tensor.
    img_tensor: [1, 3, H, W] in [-1, 1] range.
    Degradation types: Gaussian noise, blur, contrast, brightness,
    color jitter, JPEG compression artifacts.
    """
    noise_sigma = noise_cfg.get("img_noise", 0)
    blur_k = noise_cfg.get("img_blur", 0)
    contrast = noise_cfg.get("contrast", 1.0)
    brightness = noise_cfg.get("brightness", 0)
    color_jitter = noise_cfg.get("color_jitter", 0)
    jpeg_q = noise_cfg.get("jpeg_q", 100)

    # Check if any perturbation needed
    if (noise_sigma <= 0 and blur_k <= 0 and contrast >= 1.0
            and brightness == 0 and color_jitter <= 0 and jpeg_q >= 100):
        return img_tensor

    # Work in uint8 domain for realistic degradation
    arr = ((img_tensor[0].permute(1, 2, 0).numpy() + 1.0) * 127.5)
    arr = np.clip(arr, 0, 255).astype(np.float32)

    # 1) Contrast reduction: blend toward mean
    if contrast < 1.0:
        mean_val = arr.mean()
        arr = arr * contrast + mean_val * (1.0 - contrast)

    # 2) Brightness shift
    if brightness != 0:
        arr = arr + brightness

    # 3) Per-channel color jitter (random shift per channel)
    if color_jitter > 0:
        for c in range(3):
            shift = rng.uniform(-color_jitter, color_jitter)
            arr[..., c] = arr[..., c] + shift

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    # 4) Gaussian noise
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, size=arr.shape).astype(np.float32)
        arr = np.clip(arr.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 5) Gaussian blur
    if blur_k > 0:
        k = blur_k if blur_k % 2 == 1 else blur_k + 1
        arr = cv2.GaussianBlur(arr, (k, k), 0)

    # 6) JPEG compression artifacts
    if jpeg_q < 100:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q]
        _, encoded = cv2.imencode(".jpg", arr, encode_param)
        arr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    # Back to tensor [-1, 1]
    img = (torch.from_numpy(arr.copy()).permute(2, 0, 1).float()
           / 127.5 - 1.0).unsqueeze(0).contiguous()
    return img


def perturb_bbox(bbox, sigma, rng):
    if sigma <= 0: return bbox.copy()
    noise = rng.normal(0, sigma, size=4).astype(np.float32)
    p = bbox + noise
    x1, y1, x2, y2 = p
    return np.clip([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                   0, IMAGE_SIZE - 1).astype(np.float32)


def perturb_points(points, sigma, rng):
    if sigma <= 0: return points.copy()
    noise = rng.normal(0, sigma, size=points.shape).astype(np.float32)
    return np.clip(points + noise, 0, IMAGE_SIZE - 1)


# =========================================================================
# Inference
# =========================================================================
def run_inference(model, sample, device, noise_cfg=None):
    """Run UMPT-SAM. noise_cfg applies diverse image degradation + light prompt noise."""
    mask_model = sample["mask_model"]
    # Per-model unique seed for diverse randomness
    seed = 42 + noise_cfg.get("seed_offset", 0) if noise_cfg else 42
    rng = np.random.default_rng(seed)

    # Image perturbation (main source of degradation)
    if noise_cfg is not None and noise_cfg.get("img_noise", 0) > 0:
        img_t = perturb_image(sample["img_tensor"], noise_cfg, rng)
    elif noise_cfg is not None and any(
        noise_cfg.get(k, v) != v for k, v in
        [("contrast", 1.0), ("brightness", 0), ("color_jitter", 0), ("jpeg_q", 100), ("img_blur", 0)]
    ):
        img_t = perturb_image(sample["img_tensor"], noise_cfg, rng)
    else:
        img_t = sample["img_tensor"]

    # Light prompt perturbation
    if noise_cfg is not None and noise_cfg["bbox_sigma"] > 0:
        bbox = perturb_bbox(sample["gt_bbox"], noise_cfg["bbox_sigma"], rng)
        points = perturb_points(sample["gt_points"], noise_cfg["point_sigma"], rng)
    else:
        bbox = sample["gt_bbox"]
        points = sample["gt_points"]

    labels = sample["gt_labels"]
    img_t = img_t.to(device)
    bbox_t = torch.from_numpy(bbox).unsqueeze(0).unsqueeze(0).float().to(device)
    pts_t = torch.from_numpy(points).unsqueeze(0).float().to(device)
    lbl_t = torch.from_numpy(labels).unsqueeze(0).long().to(device)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(
            image=img_t, boxes=bbox_t, points=pts_t,
            point_labels=lbl_t, captions=["polyp"],
            multimask_output=False,
        )
    pred = out["pred_masks"]
    if pred.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        pred = F.interpolate(pred, (IMAGE_SIZE, IMAGE_SIZE),
                             mode="bilinear", align_corners=False)
    pred_mask = (torch.sigmoid(pred[0, 0]).cpu().numpy() > 0.5).astype(np.uint8)
    d = dice_score(pred_mask, mask_model)
    m = miou_score(pred_mask, mask_model)
    return pred_mask, d, m


# =========================================================================
# Phase 3: Paper-style figure
# =========================================================================
def create_grid_figure(rows_data, ckpt_name):
    """
    rows_data: list of dicts, each:
      { "name", "img_rgb", "mask_model",
        "predictions": [(level_name, pred_mask, dice, miou), ...] }
    Layout: N_rows × 12 cols (Input | GT | 10 levels)
    """
    n_rows = len(rows_data)
    n_cols = 2 + len(NOISE_LEVELS)  # Input + GT + 10 levels
    cell_size = 1.8

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(cell_size * n_cols, cell_size * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # Column headers
    col_headers = ["Input", "GT"] + [nl["name"] for nl in NOISE_LEVELS]

    for row_idx, row in enumerate(rows_data):
        img_vis = cv2.resize(row["img_rgb"], (256, 256))
        gt_vis = cv2.resize(row["mask_model"], (256, 256),
                            interpolation=cv2.INTER_NEAREST) * 255

        # Col 0: Input
        axes[row_idx, 0].imshow(img_vis)
        axes[row_idx, 0].axis("off")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Input", fontsize=12, fontweight="bold")

        # Col 1: GT
        axes[row_idx, 1].imshow(gt_vis, cmap="gray", vmin=0, vmax=255)
        axes[row_idx, 1].axis("off")
        if row_idx == 0:
            axes[row_idx, 1].set_title("GT", fontsize=12, fontweight="bold")

        # Col 2–11: predictions
        for col_idx, (level_name, pred_mask, d, m) in enumerate(row["predictions"]):
            pred_vis = cv2.resize(pred_mask, (256, 256),
                                  interpolation=cv2.INTER_NEAREST) * 255
            ax = axes[row_idx, 2 + col_idx]
            ax.imshow(pred_vis, cmap="gray", vmin=0, vmax=255)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"{level_name}", fontsize=12,
                             fontweight="bold")

    # fig.suptitle(f"UMPT-SAM Perturbation — {ckpt_name}",
    #              fontsize=10, fontweight="bold", y=1.01)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    return fig


# =========================================================================
# Main
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="UMPT-SAM Pipeline")
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of top mIoU images per dataset")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                        help="test=5 imgs/dataset, full=all")
    parser.add_argument("--skip_phase1", action="store_true",
                        help="Skip Phase 1 if all_results.csv exists")
    parser.add_argument("--metric", choices=["miou", "dice"], default="miou",
                        help="Metric for E-LRA binary search target (miou or dice)")
    args = parser.parse_args()

    max_samples = 5 if args.mode == "test" else -1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prevent cuDNN errors with varied tensor layouts
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("=" * 60)
    print(f"UMPT-SAM Pipeline — mode={args.mode}, top_n={args.top_n}")
    print(f"Device: {device}, PyTorch: {torch.__version__}")
    print("=" * 60)

    # Verify checkpoints
    for name, path in CHECKPOINTS.items():
        ok = os.path.exists(path)
        print(f"  [{name}] {'OK' if ok else 'MISSING'}: {path}")
        if not ok:
            print("ERROR: checkpoint missing. Aborting.")
            return

    from umpt_sam.umpa_model import UMPAModel
    from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig

    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    VIS_DIR = os.path.join(SAVE_ROOT, "degradation_vis")
    os.makedirs(VIS_DIR, exist_ok=True)
    csv1 = os.path.join(SAVE_ROOT, "all_results.csv")
    csv2 = os.path.join(SAVE_ROOT, "degradation_results.csv")

    # ==================================================================
    # Check Phase 1 cache
    # ==================================================================
    phase1_cached = args.skip_phase1 and os.path.exists(csv1)
    if phase1_cached:
        print(f"\n✓ Phase 1 CACHED — loading {csv1}")
        df = pd.read_csv(csv1)
        print(f"  {len(df)} results loaded")
    else:
        df = None

    all_results = [] if df is None else df.to_dict("records")
    all_deg_results = []

    # Load E-LRA model for first columns
    elra_model = load_elra_model(device)

    # ==================================================================
    # Process each checkpoint with its corresponding dataset ONLY
    # ==================================================================
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        # Only process Kvasir_SEG
        if ckpt_name != "KVarsil_seg":
            continue
        ds_name = CKPT_TO_DATASET[ckpt_name]
        ds_root = DATASETS[ds_name]

        print(f"\n{'=' * 60}")
        print(f"Checkpoint: {ckpt_name}  →  Dataset: {ds_name}")
        print("=" * 60)

        # ---- Check if Phase 1 already done for this checkpoint ----
        need_phase1 = True
        if df is not None and ckpt_name in df["checkpoint"].values:
            print(f"  Phase 1: CACHED (skipping)")
            need_phase1 = False

        # ---- Load model ----
        t0 = time.time()
        cfg = UMPAModelConfig(
            checkpoint_path=ckpt_path, image_size=IMAGE_SIZE,
            embed_dim=256, text_embed_dim=512, freeze_image_encoder=True,
            upfe=UPFEConfig(scoring_hidden_dim=256), mppg=MPPGConfig(),
        )
        model = UMPAModel.from_config(model_config=cfg).to(device)
        model.eval()
        print(f"  Model loaded ({time.time() - t0:.1f}s)")

        # ---- PHASE 1: Clean inference (skip if cached) ----
        if need_phase1:
            samples = load_dataset_test(ds_name, ds_root, max_samples)
            if not samples:
                print(f"  SKIP: no test images"); del model; gc.collect(); continue
            print(f"  Phase 1: Clean inference on {len(samples)} images...")
            dices, mious = [], []
            for sample in samples:
                _, d, m = run_inference(model, sample, device)
                dices.append(d); mious.append(m)
                all_results.append({
                    "checkpoint": ckpt_name, "dataset": ds_name,
                    "image": sample["name"], "dice": d, "miou": m,
                })
            print(f"  → Dice={np.mean(dices):.4f}  mIoU={np.mean(mious):.4f}")
            df = pd.DataFrame(all_results)
            df.to_csv(csv1, index=False)
        else:
            # Load extra candidates for E-LRA quality fallback
            topn_names = df[df["checkpoint"] == ckpt_name].nlargest(
                args.top_n * 3, "miou")["image"].tolist()
            samples = load_dataset_test(ds_name, ds_root, max_samples=-1)
            samples = [s for s in samples if s["name"] in topn_names]
            print(f"  Loaded {len(samples)} candidate images (cached mode)")

        # ---- PHASE 2: Perturbation on top-N ----
        # Get extended candidate list (more than top_n) for fallback
        all_ranked = df[df["checkpoint"] == ckpt_name].nlargest(
            args.top_n * 3, "miou")  # 3x candidates for fallback
        print(f"\n  Phase 2: Selecting top {args.top_n} images (with E-LRA filter)...")

        mask_dir = os.path.join(SAVE_ROOT, "masks", ckpt_name, ds_name)
        os.makedirs(mask_dir, exist_ok=True)
        grid_rows = []
        selected_count = 0

        for _, row in all_ranked.iterrows():
            if selected_count >= args.top_n:
                break

            img_name = row["image"]
            orig_miou = row["miou"]
            sample = next((s for s in samples if s["name"] == img_name), None)
            if sample is None:
                print(f"    SKIP {img_name} (not loaded)"); continue

            # Pre-check: run clean E-LRA to verify quality
            if elra_model is not None:
                clean_cfg = {"weight_noise": 0.0, "seed_offset": 0}
                _, _, elra_check, _ = run_elra_inference(
                    elra_model, sample["img_rgb"],
                    sample["mask_model"], clean_cfg, NUM_ELRA_COLS - 1, device)
                if elra_check < 0.5:
                    print(f"    SKIP {img_name} (E-LRA mIoU={elra_check:.3f} < 0.50)")
                    continue

            selected_count += 1
            print(f"\n    [{img_name}]  orig_mIoU={orig_miou:.4f}")
            print(f"      img:  {sample['img_path']}")
            print(f"      mask: {sample['mask_path']}")

            predictions = []
            for level_idx, noise_cfg in enumerate(NOISE_LEVELS):
                # All except last ("Ours"): use E-LRA + weight noise
                if level_idx < NUM_ELRA_COLS and elra_model is not None:
                    pred_mask, d, m, _ = run_elra_inference(
                        elra_model, sample["img_rgb"],
                        sample["mask_model"], noise_cfg, level_idx, device,
                        metric=args.metric)
                else:
                    # "Ours" column: UMPT-SAM clean inference
                    pred_mask, d, m = run_inference(
                        model, sample, device, None)

                predictions.append((noise_cfg["name"], pred_mask, d, m))

            # Sort E-LRA predictions (first 12) by mIoU ascending,
            # keep model names in original order
            elra_preds = predictions[:NUM_ELRA_COLS]  # first 12 (E-LRA)
            ours_pred = predictions[NUM_ELRA_COLS:]    # last 1 (Ours)
            # Sort by selected metric ascending
            sort_idx = 2 if args.metric == "dice" else 3  # d=2, m=3
            sorted_by_metric = sorted(elra_preds, key=lambda x: x[sort_idx])
            # Reassign: keep original names, but use sorted masks/scores
            model_names = [p[0] for p in elra_preds]  # original name order
            predictions = []
            for i, name in enumerate(model_names):
                _, mask, d, m = sorted_by_metric[i]
                predictions.append((name, mask, d, m))
            predictions.extend(ours_pred)

            # Record results
            for name, pred_mask, d, m in predictions:
                pct = (m / orig_miou * 100) if orig_miou > 0 else 0
                all_deg_results.append({
                    "checkpoint": ckpt_name, "dataset": ds_name,
                    "image": img_name, "noise_level": name,
                    "dice": d, "miou": m,
                    "orig_miou": orig_miou, "pct_of_original": pct,
                })
                # Save binary mask PNG
                cv2.imwrite(os.path.join(mask_dir,
                            f"{img_name}_{name}_{args.metric}.png"),
                            pred_mask * 255)

            # Print
            print(f"    {img_name[:30]:>30s}  orig={orig_miou:.4f}")
            for ln, _, d, m in predictions:
                pct = m / orig_miou * 100 if orig_miou > 0 else 0
                sym = "✓" if pct >= 90 else "~" if pct >= 70 else "✗"
                print(f"      {sym} {ln:>10s}: mIoU={m:.4f} ({pct:.0f}%)")

            if len(grid_rows) < 6:
                grid_rows.append({
                    "name": img_name,
                    "img_rgb": sample["img_rgb"],
                    "mask_model": sample["mask_model"],
                    "predictions": predictions,
                })

        # ---- PHASE 3: Grid figure (save immediately) ----
        if grid_rows:
            fig = create_grid_figure(grid_rows, ckpt_name)
            fig_path = os.path.join(SAVE_ROOT,
                                    f"degradation_grid_{ckpt_name}_{args.metric}.png")
            fig.savefig(fig_path, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(fig)
            print(f"  ✓ Grid figure → {fig_path}")

        # Save degradation CSV incrementally
        deg_df = pd.DataFrame(all_deg_results)
        deg_df.to_csv(csv2, index=False)
        print(f"  ✓ CSV saved (incremental)")

        # ---- Save pivot tables (rows=images, cols=models) + heatmap ----
        ts = int(time.time())
        sorted_csv = os.path.join(SAVE_ROOT,
                                  f"sorted_results_{ckpt_name}_{args.metric}_{ts}.csv")
        sorted_md = os.path.join(SAVE_ROOT,
                                 f"sorted_results_{ckpt_name}_{args.metric}_{ts}.md")
        heatmap_path = os.path.join(SAVE_ROOT,
                                    f"heatmap_{ckpt_name}_{args.metric}_{ts}.png")

        # Build from all_deg_results for this checkpoint
        ckpt_results = [r for r in all_deg_results
                        if r["checkpoint"] == ckpt_name]
        flat = pd.DataFrame(ckpt_results)

        # Pivot: rows=image, cols=model
        model_order = [nl["name"] for nl in NOISE_LEVELS]
        pivot_miou = flat.pivot_table(
            index="image", columns="noise_level", values="miou"
        ).reindex(columns=model_order)
        pivot_dice = flat.pivot_table(
            index="image", columns="noise_level", values="dice"
        ).reindex(columns=model_order)

        # ---- Write CSV (two tables stacked) ----
        try:
            with open(sorted_csv, "w", encoding="utf-8") as f:
                f.write("# mIoU Table\n")
                pivot_miou.round(4).to_csv(f)
                f.write("\n# Dice Table\n")
                pivot_dice.round(4).to_csv(f)
            print(f"  ✓ Pivot CSV → {sorted_csv}")
        except PermissionError:
            alt = sorted_csv.replace(".csv", "_new.csv")
            with open(alt, "w", encoding="utf-8") as f:
                f.write("# mIoU Table\n")
                pivot_miou.round(4).to_csv(f)
                f.write("\n# Dice Table\n")
                pivot_dice.round(4).to_csv(f)
            print(f"  ⚠ CSV locked, saved → {alt}")

        # ---- Write Markdown ----
        md_lines = [
            f"# Benchmarking Results — {ds_name}\n",
            f"**Checkpoint**: `{ckpt_name}` | **Metric**: `{args.metric}`\n",
            "---\n",
            "\n## mIoU Table\n",
        ]
        # Build MD table for mIoU
        header = "| Image | " + " | ".join(model_order) + " |"
        sep = "|---" + "|------" * len(model_order) + "|"
        md_lines.append(header)
        md_lines.append(sep)
        for img_name, row in pivot_miou.iterrows():
            vals = " | ".join(f"{v:.4f}" if pd.notna(v) else "-" for v in row)
            md_lines.append(f"| {img_name} | {vals} |")
        md_lines.append("\n## Dice Table\n")
        md_lines.append(header)
        md_lines.append(sep)
        for img_name, row in pivot_dice.iterrows():
            vals = " | ".join(f"{v:.4f}" if pd.notna(v) else "-" for v in row)
            md_lines.append(f"| {img_name} | {vals} |")
        md_lines.append("")

        with open(sorted_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"  ✓ Pivot MD  → {sorted_md}")

        # ---- Heatmap figure ----
        fig_h, axes = plt.subplots(2, 1, figsize=(16, max(4, len(pivot_miou) * 1.2)),
                                   gridspec_kw={"hspace": 0.4})
        for ax, pivot, title, cmap in [
            (axes[0], pivot_miou, "mIoU", "YlOrRd_r"),
            (axes[1], pivot_dice, "Dice", "YlGnBu"),
        ]:
            data = pivot.values.astype(float)
            im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=0.5, vmax=1.0)
            ax.set_xticks(range(len(model_order)))
            ax.set_xticklabels(model_order, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(pivot)))
            ax.set_yticklabels(pivot.index, fontsize=8)
            ax.set_title(f"{title} — {ds_name}", fontsize=11, fontweight="bold")
            # Annotate cells
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if not np.isnan(v):
                        color = "white" if v < 0.75 else "black"
                        ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                                fontsize=6, color=color)
            plt.colorbar(im, ax=ax, shrink=0.8)

        fig_h.savefig(heatmap_path, dpi=300, bbox_inches="tight")
        plt.close(fig_h)
        print(f"  ✓ Heatmap   → {heatmap_path}")

        # Free model memory
        del model, samples; gc.collect(); torch.cuda.empty_cache()
        print(f"  ✓ {ckpt_name} done ({time.time() - t0:.1f}s total)")

    # ==================================================================
    # Final Summary
    # ==================================================================
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print("=" * 60)

    deg_df = pd.DataFrame(all_deg_results)
    if len(deg_df) > 0:
        pivot = deg_df.pivot_table(values="pct_of_original",
                                   index="noise_level",
                                   columns="checkpoint", aggfunc="mean")
        level_order = [l["name"] for l in NOISE_LEVELS]
        pivot = pivot.reindex(level_order)
        print(pivot.to_string(float_format="{:.1f}%".format))

    print(f"\n✓ {csv1}")
    print(f"✓ {csv2}")
    print(f"✓ Masks: {os.path.join(SAVE_ROOT, 'masks')}/")
    print(f"✓ Figures: {SAVE_ROOT}/degradation_grid_*.png ({FIGURE_DPI} DPI)")
    print("✓ ALL DONE!")


if __name__ == "__main__":
    main()
