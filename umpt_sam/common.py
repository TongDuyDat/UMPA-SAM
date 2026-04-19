"""Shared utilities for UMPA-SAM training pipelines.

This module contains common classes and helpers used across
different training scripts (train_all.py, train_ablation.py, etc.).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List


# ═══════════════════════════════════════════════════════════════════════
# EpochRecord — epoch-level training metrics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EpochRecord:
    """Training metrics for a single epoch.

    Used by both full-benchmark training and ablation study training
    to track epoch-by-epoch progress in ``training_history.json``.
    """

    epoch: int
    phase: str
    lr: float
    train_total_loss: float
    train_seg_loss: float
    train_con_loss: float
    val_dice: float
    val_miou: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "phase": self.phase,
            "lr": self.lr,
            "train_total_loss": round(self.train_total_loss, 6),
            "train_seg_loss": round(self.train_seg_loss, 6),
            "train_con_loss": round(self.train_con_loss, 6),
            "val_dice": round(self.val_dice, 6),
            "val_miou": round(self.val_miou, 6),
        }


# ═══════════════════════════════════════════════════════════════════════
# ResultsManager — save/load training logs & test results
# ═══════════════════════════════════════════════════════════════════════

class ResultsManager:
    """Manage training history and test result files.

    Provides a unified interface for saving epoch-by-epoch training
    metrics and final test evaluation results as JSON files.

    Directory layout per run::

        save_dir/
        ├── training_history.json   ← epoch-by-epoch metrics
        ├── test_results.json       ← final test evaluation
        ├── run_config.json         ← hyperparameters snapshot
        ├── training_log.txt        ← readable log
        ├── best_model.pth
        └── latest_model.pth
    """

    @staticmethod
    def save_training_history(save_dir: str, history: List[EpochRecord]) -> str:
        """Save epoch-by-epoch training metrics to JSON.

        Parameters
        ----------
        save_dir : str
            Run directory (e.g. ``checkpoints/full_train/kvasir_seg/run_xxx``).
        history : list[EpochRecord]
            List of epoch records accumulated during training.

        Returns
        -------
        str
            Path to saved ``training_history.json``.
        """
        path = os.path.join(save_dir, "training_history.json")
        data = [record.to_dict() for record in history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def save_test_results(
        save_dir: str,
        metrics: Dict[str, Any],
        name: str = "",
    ) -> str:
        """Save final test evaluation metrics to JSON.

        Parameters
        ----------
        save_dir : str
            Run directory.
        metrics : dict
            Output from ``evaluate(full_metrics=True)``.
        name : str
            Dataset or scenario name for metadata.

        Returns
        -------
        str
            Path to saved ``test_results.json``.
        """
        path = os.path.join(save_dir, "test_results.json")
        data = {
            "name": name,
            "metrics": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load_training_history(save_dir: str) -> List[Dict[str, Any]]:
        """Load training_history.json from a run directory."""
        path = os.path.join(save_dir, "training_history.json")
        if not os.path.isfile(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def load_test_results(save_dir: str) -> Dict[str, Any]:
        """Load test_results.json from a run directory."""
        path = os.path.join(save_dir, "test_results.json")
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
