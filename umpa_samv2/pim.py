"""Phase 5 — Prompt Importance Module v2 (PIM v2).

Supports optional modality permutation via ``permute_order`` parameter
for permutation-invariant training (Method 2a/2b).

⭐ Core innovation of UMPAv2.  4-way importance gate + cross-modal attention.

Gate:
    [w_t, w_b, w_p, w_m] = σ(MLP(cat[pool(txt), pool(box), pool(point), pool(mask)]))

Cross-Modal Attention (each attends to the other three):
    CA_t = MHA(Q=txt,   KV=cat[box, point, mask])
    CA_b = MHA(Q=box,   KV=cat[txt, point, mask])
    CA_p = MHA(Q=point, KV=cat[txt, box, mask])
    CA_m = MHA(Q=mask,  KV=cat[txt, box, point])

Gated Residual:
    txt_enh   = LN(txt   + w_t · CA_t)
    box_enh   = LN(box   + w_b · CA_b)
    point_enh = LN(point + w_p · CA_p)
    mask_enh  = LN(mask  + w_m · CA_m)

Output:
    prompt = cat[txt_enh, box_enh, point_enh, mask_enh]  — full tokens, NOT pooled.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from .base import BasePIM


class PIMv2(BasePIM):
    """4-Way Importance Gate with Cross-Modal Attention.

    Parameters
    ----------
    embed_dim : int
        Token embedding dimension (must match all input modalities).
    num_heads : int
        Attention heads for each cross-modal MHA.
    gate_hidden_dim : int
        Hidden dim of the importance gate MLP.
    dropout : float
        Dropout in MHA layers.
    """

    NUM_MODALITIES = 4  # text, box, point, mask

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        gate_hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # ── Importance Gate MLP ──────────────────────────────────────
        # Input: cat[pool(txt), pool(box), pool(point), pool(mask)] → [B, 4D]
        # Output: [B, 4] gate logits → sigmoid → (w_t, w_b, w_p, w_m) ∈ (0, 1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.NUM_MODALITIES * embed_dim, gate_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(gate_hidden_dim, self.NUM_MODALITIES),
        )
        # Xavier init for gate MLP
        nn.init.xavier_uniform_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.xavier_uniform_(self.gate_mlp[2].weight)
        nn.init.zeros_(self.gate_mlp[2].bias)

        # ── Cross-Modal Attention (4 independent MHA) ────────────────
        self.ca_txt = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ca_box = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ca_point = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.ca_mask = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        )

        # ── Gated Residual LayerNorms ────────────────────────────────
        self.ln_txt = nn.LayerNorm(embed_dim)
        self.ln_box = nn.LayerNorm(embed_dim)
        self.ln_point = nn.LayerNorm(embed_dim)
        self.ln_mask = nn.LayerNorm(embed_dim)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _pool(x: Optional[Tensor]) -> Tensor:
        """Mean-pool over token dim: [B, N, D] → [B, D].

        Returns zeros if ``x`` is ``None``.
        """
        if x is None:
            return None
        return x.mean(dim=1)

    @staticmethod
    def _zero_pool(embed_dim: int, batch_size: int, device: torch.device) -> Tensor:
        """Return a [B, D] zero vector."""
        return torch.zeros(batch_size, embed_dim, device=device, dtype=torch.float32)

    # ── forward ──────────────────────────────────────────────────────

    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(
        self,
        txt_feats: Optional[Tensor],
        box_embs: Optional[Tensor],
        point_embs: Optional[Tensor],
        mask_embs_flat: Optional[Tensor],
        permute_order: Optional[list] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Fuse multi-modal prompt tokens with importance gating.

        Args:
            txt_feats:      ``[B, S, D]`` or ``None``.
            box_embs:       ``[B, N_box, D]`` or ``None``.
            point_embs:     ``[B, N_point, D]`` or ``None``.
            mask_embs_flat: ``[B, N_mask, D]`` (flattened dense) or ``None``.
            permute_order:  Optional list of 4 ints (e.g. ``[2,0,3,1]``)
                            specifying the concat order of enhanced tokens.
                            Gate MLP always uses canonical order.

        Returns:
            prompt_tokens: ``[B, T, D]`` — concatenated enhanced tokens.
            gate_weights:  ``[B, 4]``   — ``(w_t, w_b, w_p, w_m)`` ∈ (0, 1).
        """
        # Canonical order: [txt, box, point, mask]
        modalities = [txt_feats, box_embs, point_embs, mask_embs_flat]
        ca_modules = [self.ca_txt, self.ca_box, self.ca_point, self.ca_mask]
        ln_modules = [self.ln_txt, self.ln_box, self.ln_point, self.ln_mask]

        active = [x for x in modalities if x is not None]
        if not active:
            raise ValueError("At least one prompt modality must be provided.")
        B = active[0].shape[0]
        device = active[0].device

        # ── 1. Compute importance gate (always canonical order) ──────
        pools = []
        for x in modalities:
            if x is not None:
                pools.append(self._pool(x))
            else:
                pools.append(self._zero_pool(self.embed_dim, B, device))

        gate_input = torch.cat(pools, dim=-1)  # [B, 4D]
        gate_logits = self.gate_mlp(gate_input)  # [B, 4]
        gate_weights = torch.sigmoid(gate_logits)  # [B, 4] ∈ (0, 1)

        # Per-modality gate weights [B, 1, 1]
        ws = [gate_weights[:, i:i+1].unsqueeze(1) for i in range(4)]

        # Force gate to 0 for missing modalities
        for i, x in enumerate(modalities):
            if x is None:
                ws[i] = ws[i] * 0.0

        # ── 2. Cross-modal attention ─────────────────────────────────
        enhanced = [None] * 4
        for i in range(4):
            if modalities[i] is None:
                continue
            # KV = concat of OTHER modalities
            others = [modalities[j] for j in range(4) if j != i and modalities[j] is not None]
            kv = torch.cat(others, dim=1) if others else None
            if kv is not None:
                ca_out, _ = ca_modules[i](modalities[i], kv, kv)
                enhanced[i] = ln_modules[i](modalities[i] + ws[i] * ca_out)
            else:
                enhanced[i] = ln_modules[i](modalities[i])

        # ── 3. Concatenate enhanced tokens (with optional permutation) ─
        order = permute_order if permute_order is not None else list(range(4))
        parts = [enhanced[i] for i in order if enhanced[i] is not None]
        prompt_tokens = torch.cat(parts, dim=1)  # [B, T_total, D]

        return prompt_tokens, gate_weights

    @staticmethod
    def _cat_others(*tensors: Optional[Tensor]) -> Optional[Tensor]:
        """Concatenate non-None tensors along token dim."""
        parts = [t for t in tensors if t is not None]
        if not parts:
            return None
        return torch.cat(parts, dim=1)
