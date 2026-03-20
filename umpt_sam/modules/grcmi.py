"""Gated Residual Cross-Modal Injection (GRCMI).

Implements the injection mechanism:
    E_m' = E_m + g_m ⊙ φ_m(e_fused)

where:
    g_m  = σ(W_g^(m) · e_fused + b_g^(m))   -- gating vector, near-zero at init
    φ_m  = W_φ^(m) · e_fused + b_φ^(m)       -- modality-specific projection

Gate bias is initialised to ``gate_init_bias`` (default −3) so that
σ(−3) ≈ 0.047, ensuring the injection is near-identity at epoch 0 and
does not disturb pre-trained SAM3 weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GRCMIConfig:
    """Configuration for GRCMI module."""

    embed_dim: int = 256
    gate_init_bias: float = -3.0  # σ(-3) ≈ 0.047, near-zero gate at init
    modalities: tuple[str, ...] = ("point", "box", "mask", "text")

    def __post_init__(self) -> None:
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be > 0")


# ---------------------------------------------------------------------------
# Single-modality injector
# ---------------------------------------------------------------------------

class _ModalityGate(nn.Module):
    """Gate + projection for a single modality.

    Computes:
        g   = σ(W_g · e_fused + b_g)        -- [B, C]
        φ   = W_φ · e_fused + b_φ            -- [B, C]
        out = g ⊙ φ                          -- [B, C]

    Parameters
    ----------
    embed_dim : int
        Channel dimension C (256 for SAM).
    gate_init_bias : float
        Initial value for b_g.  W_g is zero-initialised so the initial
        gate output is σ(gate_init_bias) ≈ 0.
    """

    def __init__(self, embed_dim: int, gate_init_bias: float = -3.0) -> None:
        super().__init__()

        # Gating branch: W_g zero-init, b_g = gate_init_bias
        self.gate_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, gate_init_bias)

        # Projection branch: Xavier uniform (standard for linear layers)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)

    def forward(self, e_fused: torch.Tensor) -> torch.Tensor:
        """Compute gated projection of e_fused.

        Parameters
        ----------
        e_fused : torch.Tensor
            Fused prompt embedding, shape ``[B, C]``.

        Returns
        -------
        torch.Tensor
            Gated injection vector, shape ``[B, C]``.
        """
        # g = σ(W_g · e_fused + b_g)   -- [B, C]
        g = torch.sigmoid(self.gate_proj(e_fused))
        # φ = W_φ · e_fused + b_φ      -- [B, C]
        phi = self.value_proj(e_fused)
        # g ⊙ φ                        -- [B, C]
        return g * phi


# ---------------------------------------------------------------------------
# Full GRCMI module (all modalities)
# ---------------------------------------------------------------------------

class GatedResidualInjector(nn.Module):
    """Gated Residual Cross-Modal Injection (GRCMI).

    Injects ``e_fused`` (the UPFE output) back into each per-modality
    prompt embedding via a gated residual connection::

        E_m' = E_m + gate_m(e_fused)

    Supports:
        - **Sparse** prompts (point, box, text): ``[B, N, C]`` — the
          injection vector ``[B, 1, C]`` is broadcast across the token
          dimension to preserve positional ordering.
        - **Dense** prompt (mask): ``[B, C, H, W]`` — the injection
          vector ``[B, C, 1, 1]`` is broadcast spatially (channel-wise
          modulation).

    Parameters
    ----------
    embed_dim : int
        Embedding dimension (256 for SAM).
    gate_init_bias : float
        Bias initialisation for gates.  Default ``-3.0`` yields
        σ(−3) ≈ 0.047 so the module is near-identity at init.
    """

    SPARSE_MODALITIES = frozenset({"point", "box", "text"})
    DENSE_MODALITIES = frozenset({"mask"})

    @classmethod
    def from_config(cls, cfg: GRCMIConfig) -> "GatedResidualInjector":
        return cls(
            embed_dim=cfg.embed_dim,
            gate_init_bias=cfg.gate_init_bias,
            modalities=cfg.modalities,
        )

    def __init__(
        self,
        embed_dim: int = 256,
        gate_init_bias: float = -3.0,
        modalities: tuple[str, ...] = ("point", "box", "mask", "text"),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.modalities = modalities

        # One gate per modality — separate, not shared
        self.gates = nn.ModuleDict({
            m: _ModalityGate(embed_dim, gate_init_bias)
            for m in modalities
        })

    def inject_sparse(
        self,
        modality: str,
        embedding: torch.Tensor,
        e_fused: torch.Tensor,
    ) -> torch.Tensor:
        """Inject e_fused into a sparse embedding.

        Parameters
        ----------
        modality : str
            One of ``"point"``, ``"box"``, ``"text"``.
        embedding : torch.Tensor
            Sparse prompt embedding, shape ``[B, N, C]``.
        e_fused : torch.Tensor
            Fused embedding from UPFE, shape ``[B, C]``.

        Returns
        -------
        torch.Tensor
            Injected embedding ``E' = E + g ⊙ φ(e_fused)``, shape ``[B, N, C]``.
        """
        # gate_out: [B, C] -> [B, 1, C] for broadcast across N tokens
        gate_out = self.gates[modality](e_fused).unsqueeze(1)  # [B, 1, C]
        return embedding + gate_out  # broadcast: [B, N, C] + [B, 1, C]

    def inject_dense(
        self,
        embedding: torch.Tensor,
        e_fused: torch.Tensor,
    ) -> torch.Tensor:
        """Inject e_fused into the dense (mask) embedding.

        Parameters
        ----------
        embedding : torch.Tensor
            Dense mask embedding, shape ``[B, C, H, W]``.
        e_fused : torch.Tensor
            Fused embedding from UPFE, shape ``[B, C]``.

        Returns
        -------
        torch.Tensor
            Injected embedding, shape ``[B, C, H, W]``.
            Channel-wise modulation: E'[b,c,h,w] = E[b,c,h,w] + g[b,c]·φ(e)[b,c]
        """
        # gate_out: [B, C] -> [B, C, 1, 1] for spatial broadcast
        gate_out = self.gates["mask"](e_fused).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return embedding + gate_out  # broadcast: [B, C, H, W] + [B, C, 1, 1]

    def forward(
        self,
        embeddings: Dict[str, Optional[torch.Tensor]],
        e_fused: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Inject e_fused into all provided modality embeddings.

        Parameters
        ----------
        embeddings : dict
            Mapping from modality name to embedding tensor.  ``None``
            values are passed through unchanged.  Expected keys::

                {
                    "point": [B, N_p, C] or None,
                    "box":   [B, 2, C]   or None,
                    "mask":  [B, C, H, W] or None,
                    "text":  [B, N_t, C]  or None,
                }

        e_fused : torch.Tensor
            Fused prompt embedding from UPFE, shape ``[B, C]``.

        Returns
        -------
        dict
            Same structure as input with injected embeddings.
        """
        result: Dict[str, Optional[torch.Tensor]] = {}

        for modality, emb in embeddings.items():
            if emb is None or modality not in self.gates:
                result[modality] = emb
                continue

            if modality in self.SPARSE_MODALITIES:
                result[modality] = self.inject_sparse(modality, emb, e_fused)
            elif modality in self.DENSE_MODALITIES:
                result[modality] = self.inject_dense(emb, e_fused)
            else:
                result[modality] = emb

        return result

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, "
            f"modalities={self.modalities}, "
            f"init_gate≈{torch.sigmoid(torch.tensor(self.gates[self.modalities[0]].gate_proj.bias[0].item())):.4f}"
        )
