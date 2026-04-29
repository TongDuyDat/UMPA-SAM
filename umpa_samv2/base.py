"""Abstract base classes for swappable UMPAv2 modules.

Each ABC defines a strict interface so that alternative implementations
can be dropped in without changing the orchestrator (``UMPAv2Model``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


class BaseGeometryEncoder(nn.Module, ABC):
    """Phase 3 — Extract per-modality embeddings from raw spatial prompts.

    Implementations wrap a pretrained ``PromptEncoder`` or provide a
    custom geometry encoding pipeline.

    Note: Pipeline A does not use ``get_dense_pe`` or ``get_no_mask_embed``
    (those are Pipeline B / MaskDecoder concepts).
    """

    @abstractmethod
    def encode_boxes(self, boxes: Tensor) -> Tensor:
        """Encode bounding boxes.

        Args:
            boxes: ``[B, N_box, 4]`` in ``(x1, y1, x2, y2)`` format.

        Returns:
            Box embeddings ``[B, 2, D]`` (two corner tokens per box).
        """
        ...

    @abstractmethod
    def encode_points(
        self,
        points: Tensor,
        labels: Tensor,
        pad: bool = False,
    ) -> Tensor:
        """Encode point prompts.

        Args:
            points: ``[B, N_pt, 2]`` pixel coordinates.
            labels: ``[B, N_pt]`` (1=positive, 0=negative).
            pad:    Whether to add a padding token (when no boxes).

        Returns:
            Point embeddings ``[B, N_pt(+1), D]``.
        """
        ...

    @abstractmethod
    def encode_masks(self, masks: Tensor) -> Tensor:
        """Encode dense mask inputs.

        Args:
            masks: ``[B, 1, H, W]`` binary or soft mask.

        Returns:
            Mask embeddings ``[B, D, H_m, W_m]``.
        """
        ...


class BaseTextProjection(nn.Module, ABC):
    """Phase 4 — Project CLIP text features to model embedding dim."""

    @abstractmethod
    def forward(self, lang_feats: Tensor) -> Tensor:
        """Project text features.

        Args:
            lang_feats: ``[S, B, text_dim]`` from CLIP encoder.

        Returns:
            Projected features ``[B, S, D]``.
        """
        ...


class BasePIM(nn.Module, ABC):
    """Phase 5 — Prompt Importance Module.

    Fuses multi-modal prompt tokens via importance gating and
    cross-modal attention.  Returns full token sequences (not pooled).
    """

    @abstractmethod
    def forward(
        self,
        txt_feats: Optional[Tensor],
        box_embs: Optional[Tensor],
        point_embs: Optional[Tensor],
        mask_embs_flat: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Fuse prompt modalities.

        Args:
            txt_feats:      ``[B, S, D]`` text tokens (or ``None``).
            box_embs:       ``[B, N_box, D]`` box tokens (or ``None``).
            point_embs:     ``[B, N_point, D]`` point tokens (or ``None``).
            mask_embs_flat: ``[B, N_mask, D]`` flattened dense mask (or ``None``).

        Returns:
            prompt_tokens: ``[B, T, D]`` concatenated enhanced tokens.
            gate_weights:  ``[B, 4]``  importance weights ``(w_t, w_b, w_p, w_m)``.
        """
        ...
