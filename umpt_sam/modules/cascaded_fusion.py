import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..config.model_config import UPFEConfig


class CascadedPromptFusionEncoder(nn.Module):
    """
    Cascaded Transformer Fusion Encoder.

    Three-stage hierarchical fusion:
        Stage 1 — Spatial Fusion:    TFM1(E_point, E_box)   → E1
        Stage 2 — Shape Refinement:  TFM2(E1, E_mask)       → E2
        Stage 3 — Semantic Modulation: TFM3(E2, E_text)     → E3 = E_fused

    When a modality is missing, its stage is bypassed (identity pass-through).
    Optionally shares weights across all three stages.
    """

    @classmethod
    def from_config(
        cls,
        upfe_config: UPFEConfig,
    ) -> "CascadedPromptFusionEncoder":
        """Build the encoder from a typed UPFEConfig dataclass."""
        return cls(
            embed_dim=upfe_config.embed_dim,
            ffn_dim=upfe_config.scoring_hidden_dim,
        )

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        share_weights: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.share_weights = share_weights

        def _build_block() -> nn.TransformerDecoderLayer:
            return nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=False,
            )

        if share_weights:
            shared_block = _build_block()
            self.tfm_spatial = shared_block
            self.tfm_shape = shared_block
            self.tfm_semantic = shared_block
        else:
            # Stage 1: Spatial Fusion (point ⊗ box)
            self.tfm_spatial = _build_block()
            # Stage 2: Shape Refinement (spatial ⊗ mask)
            self.tfm_shape = _build_block()
            # Stage 3: Semantic Modulation (shape ⊗ text)
            self.tfm_semantic = _build_block()

        # Output projection: pool sequence → single vector
        self.output_norm = nn.LayerNorm(embed_dim)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool over token dimension: (B, N, D) → (B, D)."""
        return x.mean(dim=1)

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        return_weights: bool = False,
    ):
        """
        Args:
            embeddings: Dict with optional keys:
                "point_embeddings": (B, N_point, D) — point prompt tokens
                "box_embeddings":   (B, N_box, D)   — box prompt tokens
                "mask_embeddings":  (B, N_mask, D)   — flattened dense embeddings
                "text_embeddings":  (B, N_text, D)   — text/CLIP embeddings

        Returns:
            e_fused: (B, D) — fused prompt embedding
            stage_info: dict (only if return_weights=True) — per-stage activations
        """
        point_embs = embeddings.get("point_embeddings")      # point tokens
        box_embs = embeddings.get("box_embeddings")           # box tokens
        mask_embs = embeddings.get("mask_embeddings")         # dense mask tokens
        text_embs = embeddings.get("text_embeddings")         # text tokens

        # Ensure all present embeddings have correct dim
        active = [e for e in [point_embs, box_embs, mask_embs, text_embs] if e is not None]
        if not active:
            raise ValueError("At least one prompt embedding must be provided.")

        for emb in active:
            if emb.shape[-1] != self.embed_dim:
                raise ValueError(
                    f"Expected embedding dim {self.embed_dim}, got {emb.shape[-1]}. "
                    "Permute or project before passing."
                )

        stage_info = {}

        # ── Stage 1: Spatial Fusion ──────────────────────────────────────
        # Point queries Box via cross-attention for spatial reasoning.
        has_point = point_embs is not None
        has_box = box_embs is not None

        if has_point and has_box:
            # tgt=point (query), memory=box (key/value)
            e_point_attended = self.tfm_spatial(tgt=point_embs, memory=box_embs)
            # Concatenate attended points + original box for full spatial info
            e1 = torch.cat([e_point_attended, box_embs], dim=1)
        elif has_point:
            e1 = point_embs
        elif has_box:
            e1 = box_embs
        else:
            e1 = None

        if e1 is not None:
            stage_info["e1_spatial"] = e1

        # ── Stage 2: Shape Refinement ────────────────────────────────────
        if mask_embs is not None:
            if e1 is not None:
                e2 = self.tfm_shape(tgt=e1, memory=mask_embs)
            else:
                # No spatial prior — mask becomes the representation
                e2 = mask_embs
            stage_info["e2_shape"] = e2
        else:
            e2 = e1  # bypass

        # ── Stage 3: Semantic Modulation ─────────────────────────────────
        if text_embs is not None:
            if e2 is not None:
                e3 = self.tfm_semantic(tgt=e2, memory=text_embs)
            else:
                # Only text available
                e3 = text_embs
            stage_info["e3_semantic"] = e3
        else:
            e3 = e2  # bypass

        # ── Output pooling ───────────────────────────────────────────────
        e_fused = self.output_norm(self._pool(e3))

        if return_weights:
            return e_fused, stage_info
        return e_fused


# if __name__ == "__main__":
#     B, D = 2, 256
#     points = torch.randn(B, 3, D)        # 3 point tokens
#     boxes = torch.randn(B, 4, D)         # 4 box tokens
#     mask = torch.randn(B, 64 * 64, D)    # flattened mask
#     text = torch.randn(B, 10, D)          # text tokens
#
#     encoder = CascadedPromptFusionEncoder(embed_dim=D)
#     embeddings = {
#         "point_embeddings": points,
#         "box_embeddings": boxes,
#         "mask_embeddings": mask,
#         "text_embeddings": text,
#     }
#     e_fused, info = encoder(embeddings, return_weights=True)
#     print(f"e_fused: {e_fused.shape}")       # (2, 256)
#     for k, v in info.items():
#         print(f"{k}: {v.shape}")
