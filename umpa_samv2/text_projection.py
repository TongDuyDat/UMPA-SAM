"""Phase 4 — Text Projection: CLIP dim → model embed dim.

$$\\text{txt\\_feats} = \\text{Linear}_{512 \\to 256}(\\text{lang\\_feats})$$
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .base import BaseTextProjection


class LinearTextProjection(BaseTextProjection):
    """Simple linear projection from CLIP text dim to model dim.

    Initialised with Kaiming uniform (default ``nn.Linear`` init).
    Swap with MLP / adapter / LoRA for experiments.
    """

    def __init__(self, text_dim: int = 512, embed_dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, embed_dim)

    def forward(self, lang_feats: Tensor) -> Tensor:
        """Project CLIP features.

        Args:
            lang_feats: ``[S, B, text_dim]`` from ``forward_text()``.

        Returns:
            ``[B, S, embed_dim]`` — batch-first for PIM.
        """
        # [S, B, text_dim] → [B, S, text_dim] → project → [B, S, embed_dim]
        return self.proj(lang_feats.permute(1, 0, 2))
