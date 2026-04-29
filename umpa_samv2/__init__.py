"""UMPAv2 — Unified Multi-Prompt Adaptation v2 with PIM (Prompt Importance Module).

Pipeline A (DETR Grounding) architecture:
    Backbone → TextProj → PIM → Encoder 6L → Decoder 6L → Scoring + SegHead

Sibling package to ``umpt_sam`` (v1).  SAM3 Pipeline A components
(TransformerWrapper, DotProductScoring, SegmentationHead) are imported from ``sam3``.
"""

from .config import PIMv2Config, UMPAv2ModelConfig
from .model import UMPAv2Model

__all__ = [
    "PIMv2Config",
    "UMPAv2ModelConfig",
    "UMPAv2Model",
]
