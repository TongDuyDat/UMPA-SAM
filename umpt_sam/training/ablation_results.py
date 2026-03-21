"""Ablation results manager for UMPT-SAM.

Handles saving, loading, and summarizing training results across
all ablation scenarios. This file is NEW and does NOT modify any
existing code.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EpochRecord:
    """Training metrics for a single epoch."""

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


class AblationResultsManager:
    """Collect, save, and summarize ablation study results.

    Directory structure::

        base_dir/
        ├── only_box/
        │   └── run_YYYYMMDD_HHMMSS/
        │       ├── experiment_config.json
        │       ├── training_history.json
        │       ├── test_results.json
        │       ├── best_model.pth
        │       └── training_log.txt
        ├── full_model/
        │   └── run_.../
        └── ablation_summary.csv
    """

    def __init__(self, base_dir: str = "checkpoints/ablation"):
        self.base_dir = Path(base_dir)

    # ------------------------------------------------------------------
    # Save methods (called during/after training)
    # ------------------------------------------------------------------

    def save_training_history(
        self,
        save_dir: str,
        history: List[EpochRecord],
    ) -> str:
        """Save epoch-by-epoch training metrics to JSON.

        Returns
        -------
        str
            Path to saved file.
        """
        path = os.path.join(save_dir, "training_history.json")
        data = [record.to_dict() for record in history]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    def save_test_results(
        self,
        save_dir: str,
        metrics: Dict[str, Any],
        scenario_name: str = "",
    ) -> str:
        """Save final test evaluation metrics to JSON.

        Parameters
        ----------
        save_dir : str
            Run directory (e.g. ``checkpoints/ablation/only_box/run_xxx``).
        metrics : dict
            Output from ``PolypBenchmarkEvaluator.evaluate()`` or
            ``evaluate()`` function — dict with dice, miou, etc.
        scenario_name : str
            Scenario name for metadata.

        Returns
        -------
        str
            Path to saved file.
        """
        path = os.path.join(save_dir, "test_results.json")
        data = {
            "scenario": scenario_name,
            "metrics": {
                k: round(v, 6) if isinstance(v, float) else v
                for k, v in metrics.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return path

    # ------------------------------------------------------------------
    # Load & collect methods (called after all scenarios finish)
    # ------------------------------------------------------------------

    def _find_latest_run(self, scenario_dir: Path) -> Optional[Path]:
        """Find the most recent run_* directory in a scenario folder."""
        if not scenario_dir.exists():
            return None
        runs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
            key=lambda d: d.name,
            reverse=True,
        )
        return runs[0] if runs else None

    def load_test_results(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Load test_results.json for a specific scenario (latest run)."""
        scenario_dir = self.base_dir / scenario_name
        run_dir = self._find_latest_run(scenario_dir)
        if run_dir is None:
            return None
        results_path = run_dir / "test_results.json"
        if not results_path.exists():
            return None
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_training_history(self, scenario_name: str) -> Optional[List[Dict]]:
        """Load training_history.json for a specific scenario (latest run)."""
        scenario_dir = self.base_dir / scenario_name
        run_dir = self._find_latest_run(scenario_dir)
        if run_dir is None:
            return None
        history_path = run_dir / "training_history.json"
        if not history_path.exists():
            return None
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def collect_all_results(self) -> Dict[str, Dict[str, Any]]:
        """Scan all scenario directories and collect test_results.json."""
        results = {}
        if not self.base_dir.exists():
            return results
        for scenario_dir in sorted(self.base_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            data = self.load_test_results(scenario_dir.name)
            if data is not None:
                results[scenario_dir.name] = data
        return results

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    _METRIC_COLUMNS = [
        "dice", "miou", "precision", "recall", "f2", "mask_ap",
    ]

    def generate_summary_csv(self, output_path: Optional[str] = None) -> str:
        """Generate ablation_summary.csv comparing all scenarios.

        Returns
        -------
        str
            Path to the generated CSV file.
        """
        if output_path is None:
            output_path = str(self.base_dir / "ablation_summary.csv")

        all_results = self.collect_all_results()
        if not all_results:
            print("⚠️ Không tìm thấy kết quả nào.")
            return output_path

        # Get full_model metrics for delta computation
        full_metrics = {}
        if "full_model" in all_results:
            full_metrics = all_results["full_model"].get("metrics", {})

        # Write CSV
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            header = ["scenario"] + self._METRIC_COLUMNS
            if full_metrics:
                header.append("delta_dice_vs_full")
            writer.writerow(header)

            # Rows
            for scenario_name, data in sorted(all_results.items()):
                metrics = data.get("metrics", {})
                row = [scenario_name]
                for col in self._METRIC_COLUMNS:
                    val = metrics.get(col, "")
                    row.append(f"{val:.6f}" if isinstance(val, (int, float)) else str(val))
                if full_metrics:
                    full_dice = full_metrics.get("dice", 0.0)
                    this_dice = metrics.get("dice", 0.0)
                    if isinstance(full_dice, (int, float)) and isinstance(this_dice, (int, float)):
                        delta = this_dice - full_dice
                        row.append(f"{delta:+.6f}")
                    else:
                        row.append("")
                writer.writerow(row)

        print(f"✅ Bảng so sánh đã lưu tại: {output_path}")
        return output_path

    def generate_summary_table(self) -> str:
        """Generate a markdown table string for console display.

        Returns
        -------
        str
            Markdown-formatted comparison table.
        """
        all_results = self.collect_all_results()
        if not all_results:
            return "⚠️ Không tìm thấy kết quả nào."

        full_metrics = {}
        if "full_model" in all_results:
            full_metrics = all_results["full_model"].get("metrics", {})

        # Header
        cols = self._METRIC_COLUMNS
        header = "| Scenario | " + " | ".join(c.upper() for c in cols)
        if full_metrics:
            header += " | ΔDice |"
        else:
            header += " |"
        sep = "|" + "|".join(["---"] * (len(cols) + 1 + (1 if full_metrics else 0))) + "|"

        rows = [header, sep]
        for scenario_name, data in sorted(all_results.items()):
            metrics = data.get("metrics", {})
            cells = [scenario_name]
            for col in cols:
                val = metrics.get(col, "-")
                cells.append(f"{val:.4f}" if isinstance(val, (int, float)) else str(val))
            if full_metrics:
                full_dice = full_metrics.get("dice", 0.0)
                this_dice = metrics.get("dice", 0.0)
                if isinstance(full_dice, (int, float)) and isinstance(this_dice, (int, float)):
                    delta = this_dice - full_dice
                    cells.append(f"{delta:+.4f}")
                else:
                    cells.append("-")
            rows.append("| " + " | ".join(cells) + " |")

        return "\n".join(rows)
