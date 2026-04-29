"""Phase 3 — Geometry Encoder wrapping SAM3's pretrained PromptEncoder.

Zero new parameters.  Delegates to ``PromptEncoder._embed_boxes``,
``_embed_points``, ``_embed_masks`` and exposes per-modality outputs.

Note: Pipeline A uses PIM for cross-modal fusion, so we only extract
per-modality embeddings here.  ``get_dense_pe`` / ``get_no_mask_embed``
are removed — those are Pipeline B (MaskDecoder) concepts.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from sam3.sam.prompt_encoder import PromptEncoder

from .base import BaseGeometryEncoder


class SAMGeometryEncoder(BaseGeometryEncoder):
    """Thin wrapper around SAM3 ``PromptEncoder`` for per-modality extraction.

    All weights come from the pretrained SAM3 checkpoint and are **frozen**
    by default (controlled by ``UMPAv2ModelConfig.freeze_prompt_encoder``).
    """

    def __init__(self, prompt_encoder: PromptEncoder) -> None:
        super().__init__()
        # Store as a sub-module so state_dict includes it
        self.prompt_encoder = prompt_encoder

    # ── per-modality encoding ────────────────────────────────────────

    def encode_boxes(self, boxes: Tensor) -> Tensor:
        """Encode boxes via SAM3 positional encoding + corner embeddings.

        Args:
            boxes: ``[B, N_box, 4]`` xyxy format.

        Returns:
            ``[B, 2·N_box, D]`` — two corner tokens per box.
        """
        # SAM3 _embed_boxes expects [B*N_box, 4] → reshapes internally to [B*N_box, 2, 2]
        # then returns [B*N_box, 2, D].  We reshape back to [B, 2·N_box, D].
        B, N_box, _ = boxes.shape
        flat = boxes.reshape(B * N_box, 4)
        emb = self.prompt_encoder._embed_boxes(flat)  # [B*N_box, 2, D]
        return emb.reshape(B, 2 * N_box, -1)          # [B, 2·N_box, D]

    def encode_points(
        self,
        points: Tensor,
        labels: Tensor,
        pad: bool = False,
    ) -> Tensor:
        """Encode points via SAM3 positional encoding + label embeddings.

        Args:
            points: ``[B, N_pt, 2]``.
            labels: ``[B, N_pt]``.
            pad:    Add padding token when no boxes present.

        Returns:
            ``[B, N_pt(+1), D]``.
        """
        return self.prompt_encoder._embed_points(points, labels, pad=pad)

    def encode_masks(self, masks: Tensor) -> Tensor:
        """Encode masks via SAM3 conv downscaling.

        Args:
            masks: ``[B, 1, H, W]``.

        Returns:
            ``[B, D, H_m, W_m]``.
        """
        return self.prompt_encoder._embed_masks(masks)
