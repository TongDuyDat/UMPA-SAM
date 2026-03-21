"""Experiment configuration for UMPT-SAM ablation study.

Defines ExperimentConfig dataclass and pre-built ABLATION_SCENARIOS
for 10 training scenarios. This file is NEW and does NOT modify any
existing code.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(slots=True)
class ExperimentConfig:
    """Configuration for a single ablation study scenario.

    Controls which prompt types are used and which model components
    are active during training.

    Attributes
    ----------
    name : str
        Unique scenario identifier (e.g. ``"only_box"``, ``"wo_mppg"``).
    use_box : bool
        If False, bounding box prompts are set to ``None`` before forward.
    use_point : bool
        If False, point prompts (and labels) are set to ``None``.
    use_mask : bool
        If False, coarse mask prompts are set to ``None``.
    use_text : bool
        If False, text captions are set to ``None``.
    enable_mppg : bool
        If False, ``PromptPerturbation`` is forced to ``eval()`` mode
        (identity — no perturbation applied).
    enable_upfe : bool
        If False, UPFE fusion is bypassed; only raw ``sparse_embs``
        are used for the mask decoder.
    enable_mpcl : bool
        If False, consistency loss is disabled: ``K = 0``,
        ``lambda_con = 0.0`` for all phases.
    """

    name: str
    use_box: bool = True
    use_point: bool = True
    use_mask: bool = True
    use_text: bool = True
    enable_mppg: bool = True
    enable_upfe: bool = True
    enable_mpcl: bool = True

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def effective_K(self) -> int:
        """Number of perturbation runs for consistency loss."""
        return 3 if self.enable_mpcl else 0

    @property
    def effective_lambda_con(self) -> float:
        """Consistency loss weight (only active in Phase 3)."""
        return 0.5 if self.enable_mpcl else 0.0

    @property
    def active_prompts(self) -> list[str]:
        """List of active prompt type names."""
        prompts = []
        if self.use_box:
            prompts.append("box")
        if self.use_point:
            prompts.append("point")
        if self.use_mask:
            prompts.append("mask")
        if self.use_text:
            prompts.append("text")
        return prompts

    @property
    def active_components(self) -> list[str]:
        """List of active novel component names."""
        comps = []
        if self.enable_mppg:
            comps.append("MPPG")
        if self.enable_upfe:
            comps.append("UPFE")
        if self.enable_mpcl:
            comps.append("MPCL")
        return comps

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["effective_K"] = self.effective_K
        d["effective_lambda_con"] = self.effective_lambda_con
        d["active_prompts"] = self.active_prompts
        d["active_components"] = self.active_components
        return d

    def to_json(self, path: str) -> None:
        """Save config snapshot to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load config from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Remove derived fields before constructing
        for key in ("effective_K", "effective_lambda_con", "active_prompts", "active_components"):
            data.pop(key, None)
        return cls(**data)

    def __repr__(self) -> str:
        prompts = ", ".join(self.active_prompts) or "none"
        comps = ", ".join(self.active_components) or "none"
        return (
            f"ExperimentConfig(name='{self.name}', "
            f"prompts=[{prompts}], components=[{comps}])"
        )


# ======================================================================
# Pre-defined ablation scenarios
# ======================================================================

ABLATION_SCENARIOS: Dict[str, ExperimentConfig] = {
    # --- Prompt Ablation (Group A) ---
    "only_box": ExperimentConfig(
        name="only_box",
        use_point=False,
        use_mask=False,
        use_text=False,
    ),
    "only_point": ExperimentConfig(
        name="only_point",
        use_box=False,
        use_mask=False,
        use_text=False,
    ),
    "only_mask": ExperimentConfig(
        name="only_mask",
        use_box=False,
        use_point=False,
        use_text=False,
    ),
    "only_text": ExperimentConfig(
        name="only_text",
        use_box=False,
        use_point=False,
        use_mask=False,
    ),
    "box_point": ExperimentConfig(
        name="box_point",
        use_mask=False,
        use_text=False,
    ),
    "box_point_mask": ExperimentConfig(
        name="box_point_mask",
        use_text=False,
    ),

    # --- Component Ablation (Group B) ---
    "wo_mppg": ExperimentConfig(
        name="wo_mppg",
        enable_mppg=False,
    ),
    "wo_upfe": ExperimentConfig(
        name="wo_upfe",
        enable_upfe=False,
    ),
    "wo_mpcl": ExperimentConfig(
        name="wo_mpcl",
        enable_mpcl=False,
    ),

    # --- Full Model (Group C — Baseline) ---
    "full_model": ExperimentConfig(
        name="full_model",
    ),
}


def get_scenario(name: str) -> ExperimentConfig:
    """Get a pre-defined scenario by name.

    Raises
    ------
    KeyError
        If the scenario name is not found.
    """
    if name not in ABLATION_SCENARIOS:
        available = ", ".join(sorted(ABLATION_SCENARIOS.keys()))
        raise KeyError(
            f"Unknown scenario '{name}'. Available: {available}"
        )
    return ABLATION_SCENARIOS[name]


def list_scenarios() -> list[str]:
    """Return list of all available scenario names."""
    return sorted(ABLATION_SCENARIOS.keys())
