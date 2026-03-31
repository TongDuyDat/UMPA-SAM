"""Train UMPA-SAM with E_sparse + E_fused + E_text on all 5 datasets.

Architecture:
    Standard:      sparse_prompt_embs = cat([E_sparse, E_fused])
    This script:   sparse_prompt_embs = cat([E_sparse, E_fused, E_text])

Usage:
    python experiments/train_esparse_efused_etext.py
    python experiments/train_esparse_efused_etext.py --dataset kvasir_seg
    python experiments/train_esparse_efused_etext.py --dry-run
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime

import torch
import torch.optim as optim

# Ensure project root is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.config.train_config import TrainConfig, PhaseConfig
from umpt_sam.config.experiment_config import ExperimentConfig
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.losses.composer_loss import ComposerLoss
from umpt_sam.training.phase_scheduler import PhaseScheduler
from umpt_sam.training.ablation_trainer import AblationTrainer
from umpt_sam.training.ablation_results import AblationResultsManager
from umpt_sam.training.evaluate import evaluate
from umpt_sam.data.dataset_registry import list_datasets

from train_ablation import build_dataloaders
from experiments.wrappers import ESparseEFusedETextWrapper


VARIANT = "esparse_efused_etext"


def train_scenario(
    dataset_name: str,
    save_base_dir: str = f"checkpoints/{VARIANT}",
    device: str = "cuda",
    dry_run: bool = False,
    sam_checkpoint: str = "model_trained/sam3.pt",
):
    """Train E_sparse+E_fused+E_text model on a single dataset."""
    print(f"\n{'='*60}")
    print(f"🔬 E_sparse + E_fused + E_text — Dataset: {dataset_name}")
    print(f"{'='*60}\n")

    # 1. Config
    train_config = TrainConfig()
    if dry_run:
        train_config = TrainConfig(
            batch_size=2, K=train_config.K,
            phase1=PhaseConfig(name="warmup", epochs=1, lambda_con=0.0,
                               freeze_image_encoder=True, freeze_prompt_encoder=True,
                               freeze_mask_decoder=True, lr=1e-4),
            phase2=PhaseConfig(name="adaptation", epochs=1, lambda_con=0.0,
                               freeze_image_encoder=True, freeze_prompt_encoder=False,
                               freeze_mask_decoder=True, lr=5e-5),
            phase3=PhaseConfig(name="consistency", epochs=1, lambda_con=0.5,
                               freeze_image_encoder=True, freeze_prompt_encoder=False,
                               freeze_mask_decoder=False, lr=1e-5),
        )

    batch_size = train_config.batch_size if not dry_run else 2

    # 2. Data
    train_loader, val_loader, test_loader = build_dataloaders(batch_size, dry_run, dataset_name)

    # 3. Model + Wrapper
    model_config = UMPAModelConfig(
        sam_checkpoint=sam_checkpoint, embed_dim=256, text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256), mppg=MPPGConfig(),
    )
    base_model = UMPAModel.from_config(model_config=model_config).to(device)
    model = ESparseEFusedETextWrapper(base_model)

    # 4. Loss + Optimizer + Scheduler
    composer_loss = ComposerLoss(config_loss=train_config.loss_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.phase1.lr, weight_decay=1e-4)
    scheduler = PhaseScheduler(train_config=train_config)

    # 5. Trainer (reuse AblationTrainer)
    exp_cfg = ExperimentConfig(name=VARIANT)
    scenario_save_dir = os.path.join(save_base_dir, dataset_name, VARIANT)
    results_manager = AblationResultsManager(base_dir=os.path.join(save_base_dir, dataset_name))

    trainer = AblationTrainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        test_loader=test_loader, optimizer=optimizer, scheduler=scheduler,
        train_config=train_config, composer_loss=composer_loss,
        evaluate_fn=evaluate, save_dir=scenario_save_dir, device=device,
        experiment_config=exp_cfg, results_manager=results_manager,
    )

    trainer.run()
    print(f"\n✅ Hoàn tất: {dataset_name} / {VARIANT}")
    return trainer


def _clear_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description=f"UMPA-SAM: Train {VARIANT} on 5 polyp datasets",
        epilog=f"Available datasets: {', '.join(list_datasets())}",
    )
    parser.add_argument("--dataset", type=str, choices=list_datasets(), help="Only this dataset")
    parser.add_argument("--save-dir", type=str, default=f"checkpoints/{VARIANT}")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--sam-checkpoint", type=str, default="model_trained/sam3.pt")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list_datasets()
    results = []
    start_total = time.time()

    print(f"\n{'#'*70}")
    print(f"#  UMPA-SAM — {VARIANT}")
    print(f"#  Datasets:  {len(datasets)} — {', '.join(datasets)}")
    print(f"#  Save dir:  {os.path.abspath(args.save_dir)}")
    print(f"#  Dry run:   {args.dry_run}")
    print(f"#  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}\n")

    for idx, ds in enumerate(datasets, 1):
        print(f"\n[{idx}/{len(datasets)}] {ds}")
        record = {"dataset": ds, "status": "RUNNING", "elapsed_sec": 0, "error": None}
        start = time.time()

        try:
            train_scenario(ds, args.save_dir, args.device, args.dry_run, args.sam_checkpoint)
            record["status"] = "OK"
            print(f"   ✅ {ds} — THÀNH CÔNG")
        except Exception as e:
            record["status"] = "FAILED"
            record["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            error_dir = os.path.join(args.save_dir, ds, VARIANT)
            os.makedirs(error_dir, exist_ok=True)
            with open(os.path.join(error_dir, "error.log"), "w", encoding="utf-8") as f:
                f.write(f"Dataset: {ds}\nVariant: {VARIANT}\n")
                f.write(f"Time: {datetime.now().isoformat()}\nError: {e}\n\n")
                f.write(traceback.format_exc())
            print(f"   ❌ {ds} — LỖI: {record['error']}")
        finally:
            record["elapsed_sec"] = round(time.time() - start, 1)
            results.append(record)
            _clear_gpu()

    elapsed = time.time() - start_total
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)

    os.makedirs(args.save_dir, exist_ok=True)
    report_path = os.path.join(args.save_dir, "experiment_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "variant": VARIANT,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_elapsed_sec": round(elapsed, 1),
            "experiments": results,
        }, f, indent=2, ensure_ascii=False)

    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAILED")
    print(f"\n📊 {ok}/{len(results)} thành công, {fail}/{len(results)} lỗi | {h}h{m}m{s}s")
    print(f"📄 Report: {report_path}")


if __name__ == "__main__":
    main()
