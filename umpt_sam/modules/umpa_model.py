import torch
import torch.nn as nn
from typing import Dict, List, Optional

from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.mask_decoder import MaskDecoder
from sam3.sam.prompt_encoder import PromptEncoder

from .upf_enconder import UnifiedPromptFusionEncoder
from .modules import PromptPerturbation


class UMPAModel(nn.Module):
    """
    Unified Multi-Prompt Adaptation (UMPA-SAM)

    Pipeline (Section 4, Unified_Multi_Prompt_Adaptation.md)
    ---------------------------------------------------------

    Prompts: box, point, mask, text(CLIP)
          │
          ▼
    PromptPerturbation (MPPG)        § 4.2
      - BBox: δ ~ N(0,σ_B²) + γ + rotation
      - Point: ε ~ N(0,σ_P²), label flip q_flip
      - Mask:  Dilate / Erode / Warp
      - Text:  η ~ N(0,σ_T²) + synonym substitution
          │
          ▼
    SAM PromptEncoder  ──────────────────────────────┐
    SAM3VLBackbone.forward_text (CLIP)               │
          │                                          │
          ▼                                          │
    UnifiedPromptFusionEncoder (UPFE)    § 4.3       │
      w_t = softmax(h(E_t))                          │
      E_fused = Σ w_t · E_t                          │
          │                                          │
          │         SAM3VLBackbone.forward_image     │
          │              → E_img (frozen ViT)        │
          │                    │                     │
          ▼                    ▼                     │
    SAM MaskDecoder ← TwoWayTransformer(E_img, E_fused+dense_embs)
          │                                          │
          ▼                                          │
    pred_masks   (used by MPCL across K perturbations)
    """

    def __init__(
        self,
        image_encoder: SAM3VLBackbone,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        embed_dim: int = 256,
        freeze_image_encoder: bool = True,
        upfe_hidden_dim: int = 256,
        perturbation_cfg: Optional[dict] = None,
    ):
        """
        Args
        ----
        image_encoder      : SAM3VLBackbone — ViT image + CLIP text encoder
        prompt_encoder     : SAM's PromptEncoder — encodes box/point/mask → embeddings
        mask_decoder       : SAM's MaskDecoder — contains TwoWayTransformer + upsampling
        embed_dim          : model embedding dimension (256 for SAM ViT-B/L)
        freeze_image_encoder : freeze ViT weights (Phase 1 & 2 of training strategy)
        upfe_hidden_dim    : scoring network hidden dim in UPFE
        perturbation_cfg   : kwargs forwarded to PromptPerturbation (None = defaults)
        """
        super().__init__()

        # ── 1. SAM3 ViT Image Encoder ──────────────────────────────────────
        self.image_encoder = image_encoder
        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        # ── 2. SAM PromptEncoder (box / point / mask → sparse & dense embs) ─
        self.prompt_encoder = prompt_encoder

        # ── 3. SAM MaskDecoder (TwoWayTransformer is built-in here) ────────
        self.sam_mask_decoder = mask_decoder

        # ── 4. Unified Prompt Fusion Encoder (UPFE, § 4.3) ─────────────────
        self.upfe_encoder = UnifiedPromptFusionEncoder(
            embed_dim=embed_dim,
            scoting_network_hidden_dim=upfe_hidden_dim,
        )

        # ── 5. Multi-Prompt Perturbation Generator (MPPG, § 4.2) ───────────
        self.perturbation = PromptPerturbation(**(perturbation_cfg or {}))

        self.embed_dim = embed_dim

    # -----------------------------------------------------------------------
    def forward(
        self,
        image: torch.Tensor,
        # Raw (unperturbed) prompts
        boxes: Optional[torch.Tensor] = None,             # (B, 4)  xyxy
        points: Optional[torch.Tensor] = None,            # (B, N, 2)
        point_labels: Optional[torch.Tensor] = None,      # (B, N)  1=fg, 0=bg
        masks: Optional[torch.Tensor] = None,             # (B, 1, H, W) coarse mask
        # Text: either raw strings (encoded via backbone CLIP) or pre-encoded
        captions: Optional[List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,   # (B, N_txt, embed_dim)
        multimask_output: bool = False,
        return_prompt_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Single forward pass for one perturbation sample.
        For MPCL, call this K times with different perturbation seeds.

        Returns
        -------
        dict:
            "pred_masks"     : (B, num_masks, H, W)   — raw logits
            "iou_predictions": (B, num_masks)
            "prompt_weights" : (B, N_tokens, upfe_hidden) — only if requested
        """

        # ── Step 1: Perturb prompts (MPPG) ────────────────────────────────
        text_str = captions[0] if (captions and len(captions) > 0) else None
        perturbed = self.perturbation(
            bbox=boxes,
            points=points,
            point_labels=point_labels,
            mask=masks,
            text=text_str,
            text_embeddings=text_embeddings,
        )
        boxes_p        = perturbed.get("bbox",          boxes)
        points_p       = perturbed.get("points",        points)
        point_labels_p = perturbed.get("point_labels",  point_labels)
        masks_p        = perturbed.get("mask",          masks)
        text_emb_p     = perturbed.get("text_embedding", text_embeddings)
        caption_p      = perturbed.get("text",          text_str)

        # ── Step 2: SAM3 ViT Encoder → E_img (frozen) ─────────────────────
        no_grad = not any(p.requires_grad for p in self.image_encoder.parameters())
        with torch.set_grad_enabled(not no_grad):
            backbone_out = self.image_encoder.forward_image(image)

        # vision_features: (B, embed_dim, H', W')  ← last FPN level
        image_embeddings: torch.Tensor = backbone_out["vision_features"]
        image_pe = self.prompt_encoder.get_dense_pe()

        # ── Step 3: CLIP text via SAM3 language backbone ───────────────────
        #    Text embeddings already include CLIP perturbation (noise + synonyms).
        #    If raw captions are given, encode them now.
        if caption_p is not None:
            text_out   = self.image_encoder.forward_text([caption_p], device=image.device)
            # language_features shape: (embed_dim, 1, seq_len) or (seq_len, 1, embed_dim)
            # SAM3VLBackbone stores as (seq_len, B, embed_dim) — permute to (B, seq_len, embed_dim)
            text_emb_p = text_out["language_features"].permute(1, 0, 2)

        # ── Step 4: SAM PromptEncoder → sparse / dense embeddings ──────────
        point_input = (points_p, point_labels_p) if (points_p is not None) else None
        sparse_embs, dense_embs = self.prompt_encoder(
            points=point_input,
            boxes=boxes_p,
            masks=masks_p,
        )
        # sparse_embs: (B, N_sparse, embed_dim)  ← box + point tokens
        # dense_embs:  (B, embed_dim, H', W')    ← mask token map

        # ── Step 5: Build UPFE input dict ──────────────────────────────────
        #    Flatten dense_embs to token sequence for UPFE
        mask_embs_flat = dense_embs.flatten(2).permute(0, 2, 1)  # (B, H'W', embed_dim)

        upfe_input: Dict[str, Optional[torch.Tensor]] = {
            "sparse_embeddings": sparse_embs,    # box + point tokens
            "mask_embeddings":   mask_embs_flat, # dense mask tokens (flattened)
            "text_embeddings":   text_emb_p,     # CLIP text tokens
        }

        # ── Step 6: UPFE → E_fused (§ 4.3) ────────────────────────────────
        #    E_fused = Σ w_t · E_t,  w_t = softmax(h(E_t))
        upfe_out = self.upfe_encoder(upfe_input, return_weights=return_prompt_weights)
        if return_prompt_weights:
            e_fused, prompt_weights = upfe_out   # (B, embed_dim), weights
        else:
            e_fused      = upfe_out              # (B, embed_dim)
            prompt_weights = None

        # Expand E_fused as a prompt token sequence: (B, 1, embed_dim)
        e_fused_token = e_fused.unsqueeze(1)

        # ── Step 7: TwoWayTransformer + MaskDecoder ────────────────────────
        #    SAM's MaskDecoder runs internally:
        #      Z = TwoWayTransformer(E_img, [e_fused_token, dense_embs])
        #    then upsamples Z to produce mask logits.
        #
        #    We pass E_fused as sparse_prompt_embeddings so TwoWayAttn sees it
        #    as the fused unified prompt instead of per-type separate tokens.
        pred_masks, iou_predictions = self.sam_mask_decoder(
            image_embeddings=image_embeddings,     # E_img from ViT
            image_pe=image_pe,
            sparse_prompt_embeddings=e_fused_token, # E_fused from UPFE
            dense_prompt_embeddings=dense_embs,     # dense mask embedding
            multimask_output=multimask_output,
        )

        out: Dict[str, torch.Tensor] = {
            "pred_masks":       pred_masks,       # (B, num_masks, H, W)
            "iou_predictions":  iou_predictions,  # (B, num_masks)
        }
        if return_prompt_weights:
            out["prompt_weights"] = prompt_weights

        return out

    # -----------------------------------------------------------------------
    def forward_k_perturbations(
        self,
        image: torch.Tensor,
        K: int = 3,
        **prompt_kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Run K independent perturbation samples for MPCL (§ 4.4).

        Ŷ^(k) = Φ(I, P̃^(k)),  k = 1 ... K
        L_con  = Σ_{i≠j} α_ij (1 - Dice(Ŷ^(i), Ŷ^(j)))

        Returns list of K forward-pass output dicts.
        """
        return [self.forward(image, **prompt_kwargs) for _ in range(K)]
