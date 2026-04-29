"""
best_result_demo.py — UMPT-SAM Best Results Demo

Usage:
    # Quick test (5 images):
    python best_result_demo.py --mode test

    # Full run (all test sets):
    python best_result_demo.py --mode full
"""
import sys, os, gc, argparse, time
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import pandas as pd

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_ROOT)

IMAGE_SIZE = 1008
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "data_benmarks")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model_trained")
SAVE_ROOT = os.path.join(SCRIPT_DIR, "best_results")

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


# ---------------------------------------------------------------------------
# Inlined helpers (avoid albumentations import chain)
# ---------------------------------------------------------------------------
def _extract_bbox(mask_bin: np.ndarray) -> np.ndarray:
    H, W = mask_bin.shape
    ys, xs = np.where(mask_bin > 0)
    if len(xs) == 0:
        return np.array([0.0, 0.0, float(W - 1), float(H - 1)], dtype=np.float32)
    return np.array(
        [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())],
        dtype=np.float32,
    )


def _sample_points(
    mask_bin: np.ndarray, n_pos: int, n_neg: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    p = pred.astype(np.float32).flatten()
    g = gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    return float((2.0 * inter) / (p.sum() + g.sum() + 1e-8))


def miou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    p = pred.astype(np.float32).flatten()
    g = gt.astype(np.float32).flatten()
    inter = (p * g).sum()
    union = p.sum() + g.sum() - inter
    return float(inter / (union + 1e-8))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset_test(ds_name: str, ds_root: str, max_samples: int = -1):
    img_dir = os.path.join(ds_root, "images")
    mask_dir = os.path.join(ds_root, "masks")
    split_file = os.path.join(ds_root, "split", "test.txt")

    if not os.path.exists(split_file):
        print(f"  SKIP {ds_name}: no test.txt")
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
                img_path = p
                break
        mask_path = None
        for ext in [".png", ".jpg", ".bmp", ".tif"]:
            p = os.path.join(mask_dir, name + ext)
            if os.path.exists(p):
                mask_path = p
                break
        if not img_path or not mask_path:
            continue

        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        img_model = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
        img_tensor = torch.from_numpy(img_model).permute(2, 0, 1).float() / 127.5 - 1.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W]

        mask_model = cv2.resize(
            (mask_gray > 127).astype(np.uint8),
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        gt_bbox = _extract_bbox(mask_model)
        gt_points, gt_labels = _sample_points(mask_model, 3, 1, rng)

        samples.append({
            "name": name,
            "img_tensor": img_tensor,
            "img_rgb": img_rgb,
            "mask_model": mask_model,
            "gt_bbox": gt_bbox,
            "gt_points": gt_points,
            "gt_labels": gt_labels,
        })
    return samples


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def run_umpt_inference(model, sample, device):
    img_t = sample["img_tensor"].to(device)
    bbox_t = torch.from_numpy(sample["gt_bbox"]).unsqueeze(0).unsqueeze(0).float().to(device)
    pts_t = torch.from_numpy(sample["gt_points"]).unsqueeze(0).float().to(device)
    lbl_t = torch.from_numpy(sample["gt_labels"]).unsqueeze(0).long().to(device)

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
    d = dice_score(pred_mask, sample["mask_model"])
    m = miou_score(pred_mask, sample["mask_model"])
    return pred_mask, d, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="UMPT-SAM Best Results Demo")
    parser.add_argument("--mode", choices=["test", "full"], default="test",
                        help="test=5 imgs/dataset, full=all test images")
    args = parser.parse_args()

    max_samples = 5 if args.mode == "test" else -1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print(f"UMPT-SAM Best Results — mode={args.mode}")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 60)

    # Verify checkpoints exist
    for name, path in CHECKPOINTS.items():
        exists = os.path.exists(path)
        print(f"  Checkpoint {name}: {'OK' if exists else 'MISSING'} ({path})")
        if not exists:
            print("ERROR: checkpoint not found. Aborting.")
            return

    # ---- Load data ----
    print(f"\n--- Loading data (max {max_samples if max_samples > 0 else 'ALL'} per dataset) ---")
    all_data = {}
    total = 0
    for ds_name, ds_root in DATASETS.items():
        samples = load_dataset_test(ds_name, ds_root, max_samples)
        all_data[ds_name] = samples
        total += len(samples)
        print(f"  {ds_name}: {len(samples)} images")
    print(f"  Total: {total} images")

    if total == 0:
        print("ERROR: No images loaded.")
        return

    # ---- Import model ----
    from umpt_sam.umpa_model import UMPAModel
    from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig

    os.makedirs(SAVE_ROOT, exist_ok=True)
    all_results = []

    # ---- Run inference per checkpoint ----
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        print(f"\n{'=' * 60}")
        print(f"Checkpoint: {ckpt_name}")
        t0 = time.time()

        ckpt_save = os.path.join(SAVE_ROOT, ckpt_name)
        os.makedirs(ckpt_save, exist_ok=True)

        model_config = UMPAModelConfig(
            checkpoint_path=ckpt_path,
            image_size=IMAGE_SIZE,
            embed_dim=256,
            text_embed_dim=512,
            freeze_image_encoder=True,
            upfe=UPFEConfig(scoring_hidden_dim=256),
            mppg=MPPGConfig(),
        )
        model = UMPAModel.from_config(model_config=model_config).to(device)
        model.eval()
        print(f"  Model loaded on {device} ({time.time() - t0:.1f}s)")

        for ds_name, samples in all_data.items():
            if not samples:
                continue

            ds_save = os.path.join(ckpt_save, ds_name)
            os.makedirs(ds_save, exist_ok=True)

            dices, mious = [], []
            for s_idx, sample in enumerate(samples):
                pred_mask, d, m = run_umpt_inference(model, sample, device)
                dices.append(d)
                mious.append(m)
                np.save(os.path.join(ds_save, f'{sample["name"]}.npy'), pred_mask)

                all_results.append({
                    "checkpoint": ckpt_name,
                    "dataset": ds_name,
                    "image": sample["name"],
                    "dice": d,
                    "miou": m,
                })

            avg_d = np.mean(dices)
            avg_m = np.mean(mious)
            print(f"  {ds_name:>20s}: Dice={avg_d:.4f}  mIoU={avg_m:.4f}  ({len(samples)} imgs)")

        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Done ({time.time() - t0:.1f}s total)")

    # ---- Save results ----
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(SAVE_ROOT, "all_results.csv")
    df.to_csv(csv_path, index=False)

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print("=" * 60)
    pivot = df.pivot_table(values="dice", index="dataset", columns="checkpoint", aggfunc="mean")
    pivot_m = df.pivot_table(values="miou", index="dataset", columns="checkpoint", aggfunc="mean")

    for ckpt in CHECKPOINTS:
        if ckpt in pivot.columns:
            print(f"\n  [{ckpt}]")
            for ds in pivot.index:
                print(f"    {ds:>20s}: Dice={pivot.loc[ds, ckpt]:.4f}  mIoU={pivot_m.loc[ds, ckpt]:.4f}")

    print(f"\n✓ {len(df)} results saved to {csv_path}")

    # ---- Top-5 best mIoU per dataset per checkpoint ----
    TOP_K = 5
    print(f"\n{'=' * 60}")
    print(f"TOP-{TOP_K} BEST mIoU PER DATASET PER CHECKPOINT")
    print("=" * 60)
    for ckpt_name in CHECKPOINTS:
        print(f"\n  ▶ [{ckpt_name}]")
        for ds_name in DATASETS:
            subset = df[(df["checkpoint"] == ckpt_name) & (df["dataset"] == ds_name)]
            if len(subset) == 0:
                continue
            topk = subset.nlargest(TOP_K, "miou")
            print(f"\n    {ds_name}:")
            for rank, (_, row) in enumerate(topk.iterrows(), 1):
                print(f"      #{rank}  {row['image']:>40s}  mIoU={row['miou']:.4f}  Dice={row['dice']:.4f}")
        print()

    # ---- Sanity check (test mode) ----
    if args.mode == "test":
        print("\n--- SANITY CHECK ---")
        ok = True
        for _, row in df.iterrows():
            if not (0.0 <= row["dice"] <= 1.0):
                print(f"  FAIL: dice={row['dice']} out of range for {row['image']}")
                ok = False
            if not (0.0 <= row["miou"] <= 1.0):
                print(f"  FAIL: miou={row['miou']} out of range for {row['image']}")
                ok = False
            npy_path = os.path.join(SAVE_ROOT, row["checkpoint"], row["dataset"], f'{row["image"]}.npy')
            if not os.path.exists(npy_path):
                print(f"  FAIL: missing {npy_path}")
                ok = False
        if ok:
            print("  ✓ All checks passed! Code is correct.")
            print("  → Now run with --mode full for complete evaluation.")
        else:
            print("  ✗ Some checks failed. Fix before full run.")


if __name__ == "__main__":
    main()
