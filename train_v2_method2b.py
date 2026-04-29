"""Train UMPAv2 — Method 2b: SAM3-style + All-Permutation Average.

Usage::

    python train_v2_method2b.py --sam-checkpoint model_trained/sam3.pt
    python train_v2_method2b.py --n-permutations 4 --lambda-perm 0.1
    python train_v2_method2b.py --dry-run
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpa_samv2.config import UMPAv2ModelConfig
from umpa_samv2.model import UMPAv2Model
from umpa_samv2.losses_matched import UMPAv2MatchedLoss
from umpa_samv2.losses_permutation import AllPermutationLoss

from umpa_samv2.training.config_v2 import TrainV2AllPermConfig
from umpa_samv2.training.trainer_v2_allperm import UMPAv2AllPermTrainer
from umpa_samv2.training.evaluate_v2 import evaluate

from umpt_sam.data.polyp_dataset import PolypDataset, collate_fn
from umpt_sam.data.dataset_registry import get_dataset_config, list_datasets
from umpt_sam.data.polyp_transforms import POLYP_TRANSFORMS


def build_dataloaders(dataset_name, batch_size, num_workers=2, dry_run=False):
    ds_cfg = get_dataset_config(dataset_name)
    train_ds = PolypDataset(cfg=ds_cfg, phase="train")
    train_ds.transform = POLYP_TRANSFORMS["train"]
    val_ds = PolypDataset(cfg=ds_cfg, phase="val")
    val_ds.transform = POLYP_TRANSFORMS["val"]
    test_ds = PolypDataset(cfg=ds_cfg, phase="test")
    test_ds.transform = POLYP_TRANSFORMS["val"]

    if dry_run:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, range(min(4, len(train_ds))))
        val_ds = Subset(val_ds, range(min(4, len(val_ds))))
        test_ds = Subset(test_ds, range(min(4, len(test_ds))))

    kw = dict(batch_size=batch_size, collate_fn=collate_fn,
              num_workers=num_workers, pin_memory=True)
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
    )


def train_single_dataset(dataset_name, save_base_dir, device, dry_run, args):
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {dataset_name} (Method 2b — All-Perm Average)")
    print(f"{'='*60}\n")

    t0 = time.time()

    cfg = TrainV2AllPermConfig(
        total_epochs=2 if dry_run else args.total_epochs,
        warmup_epochs=1 if dry_run else args.warmup_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        lr_schedule=args.lr_schedule,
        transformer_lr_mult=args.transformer_lr_mult,
        consistency_weight=args.lambda_con,
        importance_reg_weight=args.lambda_reg,
        K=args.K,
        batch_size=2 if dry_run else args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        amp_enabled=not args.no_amp,
        amp_dtype=args.amp_dtype,
        grad_clip_norm=args.grad_clip,
        # Method 2b specific
        n_permutations=args.n_permutations,
        lambda_perm=args.lambda_perm,
        perm_temperature=args.perm_temperature,
        perm_grad_mode=args.perm_grad_mode,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset_name, cfg.batch_size,
        num_workers=args.num_workers, dry_run=dry_run,
    )

    model_cfg = UMPAv2ModelConfig(
        sam_checkpoint=args.sam_checkpoint,
        embed_dim=args.embed_dim,
        text_embed_dim=args.text_embed_dim,
    )
    model = UMPAv2Model.from_config(model_cfg).to(device)

    total_p = sum(p.numel() for p in model.parameters())
    print(f"📊 Params: {total_p/1e6:.1f}M total")

    matched_loss = UMPAv2MatchedLoss(
        focal_alpha=cfg.focal_alpha,
        focal_gamma=cfg.focal_gamma,
        loss_ce_weight=cfg.loss_ce_weight,
        loss_bbox_weight=cfg.loss_bbox_weight,
        loss_giou_weight=cfg.loss_giou_weight,
        loss_mask_weight=cfg.loss_mask_weight,
        loss_dice_weight=cfg.loss_dice_weight,
        consistency_weight=cfg.consistency_weight,
        importance_reg_weight=cfg.importance_reg_weight,
    ).to(device)

    all_perm_loss = AllPermutationLoss(
        matched_loss=matched_loss,
        n_permutations=cfg.n_permutations,
        lambda_perm=cfg.lambda_perm,
        perm_temperature=cfg.perm_temperature,
        perm_grad_mode=cfg.perm_grad_mode,
    ).to(device)

    dataset_dir = os.path.join(save_base_dir, dataset_name)
    trainer = UMPAv2AllPermTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
        all_perm_loss=all_perm_loss,
        evaluate_fn=evaluate,
        save_dir=dataset_dir,
        device=device,
    )

    cfg_path = os.path.join(trainer.save_dir, "run_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(
            {"method": "method2b_all_perm", "dataset": dataset_name,
             "config": cfg.to_dict(), "timestamp": datetime.now().isoformat()},
            f, indent=2,
        )

    final_metrics = trainer.run()
    elapsed = time.time() - t0
    print(f"\n✅ {dataset_name} | Best Dice: {trainer.best_val_dice:.4f} | {elapsed:.0f}s")

    return {
        "status": "OK",
        "best_dice": trainer.best_val_dice,
        "test_metrics": final_metrics,
        "save_dir": trainer.save_dir,
        "elapsed": elapsed,
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="UMPAv2 Method 2b — SAM3-style + All-Permutation Average",
    )
    p.add_argument("--datasets", nargs="+", default=None, choices=list_datasets())
    p.add_argument("--sam-checkpoint", default="model_trained/sam3.pt")
    p.add_argument("--save-dir", default="checkpoints/umpa_v2_method2b")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--text-embed-dim", type=int, default=256)

    # Schedule
    p.add_argument("--total-epochs", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--lr-schedule", default="cosine", choices=["cosine", "step", "linear"])
    p.add_argument("--transformer-lr-mult", type=float, default=0.1)

    # Loss
    p.add_argument("--K", type=int, default=3)
    p.add_argument("--lambda-con", type=float, default=0.5)
    p.add_argument("--lambda-reg", type=float, default=0.1)

    # Method 2b specific
    p.add_argument("--n-permutations", type=int, default=6)
    p.add_argument("--lambda-perm", type=float, default=0.1)
    p.add_argument("--perm-temperature", type=float, default=1.0)
    p.add_argument("--perm-grad-mode", default="canonical_only",
                    choices=["canonical_only", "all"])

    # Runtime
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--amp-dtype", default="float16", choices=["float16", "bfloat16"])
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dry-run", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("⚠️ CUDA not available, using CPU")
        device = "cpu"

    datasets = args.datasets or list_datasets()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║   UMPAv2 METHOD 2b — ALL-PERMUTATION AVERAGE               ║
╠══════════════════════════════════════════════════════════════╣
║  Datasets:      {', '.join(datasets):<43s}║
║  N permutations: {str(args.n_permutations):<42s}║
║  λ_perm:        {str(args.lambda_perm):<43s}║
║  Dry-run:       {str(args.dry_run):<43s}║
╚══════════════════════════════════════════════════════════════╝
""")

    os.makedirs(args.save_dir, exist_ok=True)
    all_results = {}
    total_start = time.time()

    for i, ds_name in enumerate(datasets, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(datasets)}] {ds_name}")
        print(f"{'#'*60}")

        try:
            result = train_single_dataset(
                ds_name, args.save_dir, device, args.dry_run, args,
            )
        except Exception as e:
            result = {"status": "FAILED", "error": str(e)}
            print(f"\n❌ {ds_name} FAILED: {e}")
            import traceback; traceback.print_exc()
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_results[ds_name] = result

    total_elapsed = time.time() - total_start
    h, rem = divmod(int(total_elapsed), 3600)
    m, s = divmod(rem, 60)
    print(f"\n⏱️ Total: {h}h {m:02d}m {s:02d}s")

    results_path = os.path.join(args.save_dir, "all_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    failed = [k for k, v in all_results.items() if v["status"] != "OK"]
    if failed:
        print(f"\n⚠️ {len(failed)} failed: {', '.join(failed)}")
        sys.exit(1)
    print("\n🎉 All datasets trained!")


if __name__ == "__main__":
    main()
