"""CLI entry point for ablation study training.

Usage
-----
    # Train a single scenario
    python train_ablation.py --scenario only_box

    # Train all scenarios sequentially
    python train_ablation.py --all

    # Custom save directory
    python train_ablation.py --scenario wo_mppg --save-dir /workspace/results

    # Dry-run (1 batch, 1 epoch) to verify setup
    python train_ablation.py --scenario full_model --dry-run
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from umpt_sam.config.experiment_config import (
    ExperimentConfig,
    get_scenario,
    list_scenarios,
    ABLATION_SCENARIOS,
)
from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig
from umpt_sam.config.train_config import TrainConfig
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.ablation_wrapper import AblationModelWrapper
from umpt_sam.losses.composer_loss import ComposerLoss
from umpt_sam.training.phase_scheduler import PhaseScheduler
from umpt_sam.training.ablation_trainer import AblationTrainer
from umpt_sam.training.ablation_results import AblationResultsManager
from umpt_sam.training.evaluate import evaluate
from umpt_sam.data.polyp_dataset import PolypDataset, DatasetConfig, collate_fn
from umpt_sam.data.dataset_registry import get_dataset_config, list_datasets
from umpt_sam.data.polyp_transforms import POLYP_TRANSFORMS


def build_dataloaders(batch_size: int, dry_run: bool = False, dataset_name: str = "kvasir_seg"):
    """Build train/val/test dataloaders."""
    dataset_config = get_dataset_config(dataset_name)

    train_dataset = PolypDataset(cfg=dataset_config, phase='train')
    train_dataset.transform = POLYP_TRANSFORMS['train']

    val_dataset = PolypDataset(cfg=dataset_config, phase='val')
    val_dataset.transform = POLYP_TRANSFORMS['val']

    test_dataset = PolypDataset(cfg=dataset_config, phase='test')
    test_dataset.transform = POLYP_TRANSFORMS['val']

    if dry_run:
        # Use tiny subset for dry-run
        from torch.utils.data import Subset
        train_dataset = Subset(train_dataset, range(min(4, len(train_dataset))))
        val_dataset = Subset(val_dataset, range(min(4, len(val_dataset))))
        test_dataset = Subset(test_dataset, range(min(4, len(test_dataset))))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    total = sum(
        len(d.dataset) if hasattr(d, 'dataset') else len(d)
        for d in [train_loader, val_loader, test_loader]
    )
    print(f"📦 [{dataset_name}] Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    return train_loader, val_loader, test_loader


def train_scenario(
    exp_cfg: ExperimentConfig,
    save_base_dir: str = "checkpoints/ablation",
    device: str = "cuda",
    dry_run: bool = False,
    sam_checkpoint: str = "sam3.pt",
    dataset_name: str = "kvasir_seg",
):
    """Train a single ablation scenario."""
    print(f"\n{'='*60}")
    print(f"🔬 ABLATION: {exp_cfg.name}")
    print(f"   Prompts: {exp_cfg.active_prompts}")
    print(f"   Components: {exp_cfg.active_components}")
    print(f"   K={exp_cfg.effective_K}, λ_con={exp_cfg.effective_lambda_con}")
    print(f"{'='*60}\n")

    # 1. Config
    train_config = TrainConfig()
    if dry_run:
        # Override for quick test
        from umpt_sam.config.train_config import PhaseConfig
        train_config = TrainConfig(
            batch_size=2,
            K=train_config.K,
            phase1=PhaseConfig(name="warmup", epochs=1, lambda_con=0.0,
                               freeze_image_encoder=True, freeze_prompt_encoder=True,
                               freeze_mask_decoder=True, lr=1e-4),
            phase2=PhaseConfig(name="adaptation", epochs=0, lambda_con=0.0,
                               freeze_image_encoder=True, freeze_prompt_encoder=False,
                               freeze_mask_decoder=True, lr=5e-5),
            phase3=PhaseConfig(name="consistency", epochs=0, lambda_con=0.5,
                               freeze_image_encoder=True, freeze_prompt_encoder=False,
                               freeze_mask_decoder=False, lr=1e-5),
        )

    batch_size = train_config.batch_size if not dry_run else 2

    # 2. Data
    train_loader, val_loader, test_loader = build_dataloaders(batch_size, dry_run, dataset_name)

    # 3. Model
    model_config = UMPAModelConfig(
        sam_checkpoint=sam_checkpoint,
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256),
        mppg=MPPGConfig(),
    )
    base_model = UMPAModel.from_config(model_config=model_config).to(device)

    # 4. Wrap model with ablation controls
    model = AblationModelWrapper.wrap(base_model, exp_cfg)

    # 5. Loss + Optimizer + Scheduler
    composer_loss = ComposerLoss(config_loss=train_config.loss_weights).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.phase1.lr, weight_decay=1e-4)
    scheduler = PhaseScheduler(train_config=train_config)

    # 6. Results manager
    scenario_save_dir = os.path.join(save_base_dir, dataset_name, exp_cfg.name)
    results_manager = AblationResultsManager(base_dir=os.path.join(save_base_dir, dataset_name))

    # 7. Trainer
    trainer = AblationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        train_config=train_config,
        composer_loss=composer_loss,
        evaluate_fn=evaluate,
        save_dir=scenario_save_dir,
        device=device,
        experiment_config=exp_cfg,
        results_manager=results_manager,
    )

    # 8. Run training
    trainer.run()
    print(f"\n✅ Hoàn tất kịch bản: {exp_cfg.name}")
    print(f"   Kết quả: {trainer.save_dir}")

    return trainer


def main():
    parser = argparse.ArgumentParser(
        description="UMPT-SAM Ablation Study Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available scenarios: {', '.join(list_scenarios())}",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        type=str,
        choices=list_scenarios(),
        help="Scenario name to train",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Train all 10 scenarios sequentially",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all available scenarios and exit",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/ablation",
        help="Base directory for saving results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test with 1 epoch, 4 samples",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=str,
        default="sam3.pt",
        help="Path to SAM3 pre-trained checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kvasir_seg",
        choices=list_datasets(),
        help="Polyp dataset name",
    )

    args = parser.parse_args()

    if args.list:
        print("📋 Available ablation scenarios:\n")
        for name in list_scenarios():
            cfg = get_scenario(name)
            print(f"  {name:20s} | Prompts: {cfg.active_prompts} | Components: {cfg.active_components}")
        return

    if args.all:
        scenarios = [get_scenario(name) for name in list_scenarios()]
        print(f"🚀 Training {len(scenarios)} scenarios sequentially...\n")
        for i, exp_cfg in enumerate(scenarios, 1):
            print(f"\n{'#'*60}")
            print(f"# Scenario {i}/{len(scenarios)}: {exp_cfg.name}")
            print(f"{'#'*60}")
            train_scenario(
                exp_cfg=exp_cfg,
                save_base_dir=args.save_dir,
                device=args.device,
                dry_run=args.dry_run,
                sam_checkpoint=args.sam_checkpoint,
                dataset_name=args.dataset,
            )
        # Generate summary
        results_mgr = AblationResultsManager(base_dir=os.path.join(args.save_dir, args.dataset))
        csv_path = results_mgr.generate_summary_csv()
        print(f"\n📊 Summary CSV: {csv_path}")
        print(results_mgr.generate_summary_table())
    else:
        exp_cfg = get_scenario(args.scenario)
        train_scenario(
            exp_cfg=exp_cfg,
            save_base_dir=args.save_dir,
            device=args.device,
            dry_run=args.dry_run,
            sam_checkpoint=args.sam_checkpoint,
            dataset_name=args.dataset,
        )


if __name__ == "__main__":
    main()
