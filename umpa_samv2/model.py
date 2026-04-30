"""UMPAv2 Model — Pipeline A (DETR Grounding) orchestrator.

All sub-modules are swappable via abstract base classes.
SAM3 backbone, TransformerEncoder/Decoder, SegHead, DotProductScoring
are reused with pretrained weights.

Forward flow:
    1. perturbation(prompts)                       → perturbed prompts
    2. image_encoder.forward_image(img)             → backbone_out       [FROZEN]
    3. image_encoder.forward_text(caps)             → lang_feats         [FROZEN]
    4. geometry_encoder.encode_*()                  → box/point/mask_embs[FROZEN]
    5. text_projection(lang_feats)                  → txt_feats          [NEW]
    6. pim(txt, spatial, mask_flat)                  → prompt_tokens      [NEW ⭐]
    7. _build_prompt_for_encoder(pim_output)         → prompt [T,B,D]     [reshape]
    8. transformer.encoder(img_feats, prompt)        → memory             [FINE-TUNE]
    9. transformer.decoder(queries, memory, prompt)  → hs, ref_boxes      [FINE-TUNE]
   10. dot_prod_scoring(hs, prompt, prompt_mask)     → pred_logits        [FINE-TUNE]
   11. segmentation_head(hs, backbone_fpn)           → pred_masks         [FINE-TUNE]
   12. _select_best_mask(logits, masks)              → selected output    [NEW]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from sam3.model.model_misc import DotProductScoring, TransformerWrapper
from sam3.model.maskformer_segmentation import UniversalSegmentationHead
from sam3.model.vl_combiner import SAM3VLBackbone
from sam3.sam.prompt_encoder import PromptEncoder

from umpt_sam.modules.modules import PromptPerturbation

from .base import BaseGeometryEncoder, BasePIM, BaseTextProjection
from .config import UMPAv2ModelConfig
from .geometry_encoder import SAMGeometryEncoder
from .pim import PIMv2
from .text_projection import LinearTextProjection

logger = logging.getLogger(__name__)


class UMPAv2Model(nn.Module):
    """Unified Multi-Prompt Adaptation v2 — Pipeline A (DETR Grounding).

    All components are injected via constructor for easy swapping.
    Use ``from_config()`` for standard construction from SAM3 checkpoint.
    """

    # ── Construction ─────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: UMPAv2ModelConfig,
        *,
        checkpoint_path: Optional[str] = None,
        map_location: str = "cpu",
    ) -> "UMPAv2Model":
        """Build UMPAv2Model from config.

        If ``checkpoint_path`` is given, loads trained UMPAv2 weights.
        Otherwise loads SAM3 pretrained weights for fine-tuning.
        """
        ckpt = checkpoint_path or config.checkpoint_path

        # Load SAM3 pretrained model to extract all Pipeline A components
        components = cls._load_sam3_components(config)

        image_encoder = components["image_encoder"]
        prompt_encoder = components["prompt_encoder"]
        transformer = components["transformer"]
        dot_prod_scoring = components["dot_prod_scoring"]
        segmentation_head = components["segmentation_head"]

        # Build UMPAv2-specific modules
        geometry_encoder = SAMGeometryEncoder(prompt_encoder)
        text_projection = LinearTextProjection(
            text_dim=config.text_embed_dim,
            embed_dim=config.embed_dim,
        )
        pim = PIMv2(
            embed_dim=config.embed_dim,
            num_heads=config.pim.num_heads,
            gate_hidden_dim=config.pim.gate_hidden_dim,
            dropout=config.pim.dropout,
        )
        perturbation = PromptPerturbation(**config.perturbation_kwargs())

        model = cls(
            image_encoder=image_encoder,
            transformer=transformer,
            dot_prod_scoring=dot_prod_scoring,
            segmentation_head=segmentation_head,
            geometry_encoder=geometry_encoder,
            text_projection=text_projection,
            pim=pim,
            perturbation=perturbation,
            config=config,
        )

        if ckpt is not None:
            model.load_weights(ckpt, map_location=map_location)

        # SAM3 build may set default dtype to bfloat16, contaminating
        # all new modules (PIM, TextProjection).  Normalize everything
        # to float32 — autocast handles mixed-precision during forward.
        model.float()

        return model

    @classmethod
    def _load_sam3_components(cls, config: UMPAv2ModelConfig) -> Dict:
        """Load SAM3 with pretrained weights and extract Pipeline A components."""
        from sam3 import build_sam3_image_model

        sam_model = build_sam3_image_model(
            checkpoint_path=str(config.sam_checkpoint_path),
            device="cpu",
            eval_mode=False,
            enable_segmentation=True,
            enable_inst_interactivity=True,
        )

        # Pipeline A components from Sam3Image
        image_encoder = sam_model.backbone
        transformer = sam_model.transformer          # TransformerWrapper(encoder+decoder)
        dot_prod_scoring = sam_model.dot_prod_scoring
        segmentation_head = sam_model.segmentation_head

        # PromptEncoder from tracker (for SAMGeometryEncoder)
        tracker_model = getattr(
            getattr(sam_model, "inst_interactive_predictor", None), "model", None
        )
        if tracker_model is None:
            raise RuntimeError(
                "SAM3 tracker model not found — cannot extract PromptEncoder."
            )
        prompt_encoder = tracker_model.sam_prompt_encoder

        return {
            "image_encoder": image_encoder,
            "transformer": transformer,
            "dot_prod_scoring": dot_prod_scoring,
            "segmentation_head": segmentation_head,
            "prompt_encoder": prompt_encoder,
        }

    # ── Init ─────────────────────────────────────────────────────────

    def __init__(
        self,
        image_encoder: SAM3VLBackbone,
        transformer: TransformerWrapper,
        dot_prod_scoring: DotProductScoring,
        segmentation_head: UniversalSegmentationHead,
        geometry_encoder: BaseGeometryEncoder,
        text_projection: BaseTextProjection,
        pim: BasePIM,
        perturbation: PromptPerturbation,
        config: Optional[UMPAv2ModelConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # SAM3 Pipeline A components (pretrained)
        self.image_encoder = image_encoder
        self.transformer = transformer      # .encoder (6L) + .decoder (6L)
        self.dot_prod_scoring = dot_prod_scoring
        self.segmentation_head = segmentation_head

        # UMPAv2 components (new / wrapped)
        self.geometry_encoder = geometry_encoder
        self.text_projection = text_projection
        self.pim = pim
        self.perturbation = perturbation

        # Apply freeze
        if config is not None:
            self._apply_freeze(config)

    def _apply_freeze(self, config: UMPAv2ModelConfig) -> None:
        """Freeze components per config."""
        freeze_map = {
            "image_encoder": config.freeze_image_encoder,
            "geometry_encoder": config.freeze_prompt_encoder,
            "transformer.encoder": config.freeze_transformer_encoder,
            "transformer.decoder": config.freeze_transformer_decoder,
            "segmentation_head": config.freeze_segmentation_head,
            "dot_prod_scoring": config.freeze_dot_prod_scoring,
        }
        for name, should_freeze in freeze_map.items():
            if should_freeze:
                module = self
                for attr in name.split("."):
                    module = getattr(module, attr)
                for p in module.parameters():
                    p.requires_grad = False
                logger.info("Frozen: %s", name)

    # ── Forward ──────────────────────────────────────────────────────

    def forward(
        self,
        image: Tensor,
        boxes: Optional[Tensor] = None,
        points: Optional[Tensor] = None,
        point_labels: Optional[Tensor] = None,
        masks: Optional[Tensor] = None,
        captions: Optional[List[str]] = None,
        return_gate_weights: bool = False,
        return_all_queries: bool = False,
        enable_permutation: bool = False,
        permute_order: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """Full UMPAv2 Pipeline A forward pass.

        Args:
            image:           ``[B, 3, H, W]``.
            boxes:           ``[B, N_box, 4]`` xyxy (optional).
            points:          ``[B, N_pt, 2]`` (optional).
            point_labels:    ``[B, N_pt]`` (optional).
            masks:           ``[B, 1, H, W]`` (optional).
            captions:        List of strings (optional).
            return_gate_weights: Whether to return PIM gate weights.
            return_all_queries: Whether to return all 200 query masks.

        Returns:
            Dict with ``pred_masks``, ``pred_logits``, ``pred_boxes``,
            and optionally ``gate_weights``, ``all_pred_masks``.

        If ``enable_permutation`` is True (training only), generates a
        random modality order for PIM concatenation.
        If ``permute_order`` is provided, uses that explicit order.
        """
        B = image.shape[0]

        # ── Phase 1: Perturbation (train only) ───────────────────────
        single_caption = captions[0] if captions and len(captions) == 1 else None
        perturbed = self.perturbation(
            bbox=boxes,
            points=points,
            point_labels=point_labels,
            mask=masks,
            text=single_caption,
        )
        boxes_p = perturbed.get("bbox", boxes)
        points_p = perturbed.get("points", points)
        labels_p = perturbed.get("point_labels", point_labels)
        masks_p = perturbed.get("mask", masks)
        if captions is not None and "text" in perturbed:
            captions_p = [perturbed["text"]] if len(captions) == 1 else captions
        else:
            captions_p = captions

        # ── Phase 2: Backbone encoding (frozen) ──────────────────────
        with torch.no_grad():
            backbone_out = self.image_encoder.forward_image(image)

        # ── Phase 2b: Text encoding (frozen) ─────────────────────────
        lang_feats = None
        if captions_p is not None:
            with torch.no_grad():
                text_out = self.image_encoder.forward_text(
                    captions_p, device=image.device,
                )
            lang_feats = text_out["language_features"]  # [S, B, 512]

        # ── Phase 3: Geometry encoding (frozen) ──────────────────────
        box_embs = None
        point_embs = None
        mask_embs = None

        if boxes_p is not None:
            if boxes_p.ndim == 2:
                boxes_p = boxes_p.unsqueeze(1)  # [B, 4] → [B, 1, 4]
            box_embs = self.geometry_encoder.encode_boxes(boxes_p)  # [B, 2·N, D]

        if points_p is not None:
            point_embs = self.geometry_encoder.encode_points(
                points_p, labels_p, pad=(boxes_p is None),
            )  # [B, N_pt(+1), D]

        # Prepare mask for geometry encoder
        masks_for_enc = self._prepare_mask_prompt(masks_p)
        if masks_for_enc is not None:
            mask_embs = self.geometry_encoder.encode_masks(masks_for_enc)  # [B, D, H_m, W_m]

        # ── Phase 4: Text projection (skip if CLIP dim == embed_dim) ────
        txt_feats = None
        if lang_feats is not None:
            # lang_feats: [S, B, text_dim] → [B, S, text_dim]
            txt_feats_bf = lang_feats.permute(1, 0, 2)
            if txt_feats_bf.shape[-1] != self.pim.embed_dim:
                txt_feats = self.text_projection(lang_feats)  # [B, S, D]
            else:
                txt_feats = txt_feats_bf  # already correct dim, skip projection

        # ── Normalize dtype ──────────────────────────────────────────
        # Frozen SAM3 backbone may store weights in bfloat16, producing
        # bfloat16 outputs.  Under autocast(float16), PIM weights become
        # float16.  Cast all frozen outputs to float32 so autocast can
        # uniformly downcast to the target AMP dtype.
        if box_embs is not None:
            box_embs = box_embs.float()
        if point_embs is not None:
            point_embs = point_embs.float()
        if txt_feats is not None:
            txt_feats = txt_feats.float()

        # ── Phase 5: PIM v2 (trainable ⭐) ───────────────────────────
        # Flatten mask for PIM attention
        mask_embs_flat = None
        if mask_embs is not None:
            mask_embs_flat = mask_embs.float().flatten(2).permute(0, 2, 1)  # [B, H_m*W_m, D]

        # Determine permutation order
        _permute_order = permute_order
        if _permute_order is None and enable_permutation and self.training:
            _permute_order = torch.randperm(4).tolist()

        prompt_tokens, gate_weights = self.pim(
            txt_feats=txt_feats,
            box_embs=box_embs,
            point_embs=point_embs,
            mask_embs_flat=mask_embs_flat,
            permute_order=_permute_order,
        )  # [B, T, D], [B, 4]

        # ── Phase 7: Build prompt for encoder/decoder ────────────────
        # PIM outputs [B, T, D] batch-first → convert to [T, B, D] seq-first
        prompt = prompt_tokens.permute(1, 0, 2)       # [T, B, D]
        prompt_mask = torch.zeros(
            B, prompt.shape[0], dtype=torch.bool, device=image.device,
        )  # [B, T] — no padding

        # ── Phase 8: Transformer Encoder (fine-tuned) ────────────────
        # Get image features in the format encoder expects
        feat_tuple = self._get_img_feats(backbone_out)
        img_feats, img_pos_embeds, vis_feat_sizes = feat_tuple

        prompt_pos_embed = torch.zeros_like(prompt)
        memory_out = self.transformer.encoder(
            src=img_feats.copy(),
            src_key_padding_mask=None,
            src_pos=img_pos_embeds.copy(),
            prompt=prompt,
            prompt_pos=prompt_pos_embed,
            prompt_key_padding_mask=prompt_mask,
            feat_sizes=vis_feat_sizes,
        )
        # memory_out is a dict with: memory, padding_mask, pos_embed,
        # memory_text, level_start_index, spatial_shapes, valid_ratios

        # ── Phase 9: DETR Decoder (fine-tuned) ───────────────────────
        memory = memory_out["memory"]
        bs = memory.shape[1]
        query_embed = self.transformer.decoder.query_embed.weight
        tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)  # [N_q, B, D]

        apply_dac = self.transformer.decoder.dac and self.training
        hs, reference_boxes, dec_presence_out, _ = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            memory_key_padding_mask=memory_out["padding_mask"],
            pos=memory_out["pos_embed"],
            reference_boxes=None,
            level_start_index=memory_out["level_start_index"],
            spatial_shapes=memory_out["spatial_shapes"],
            valid_ratios=memory_out["valid_ratios"],
            tgt_mask=None,
            memory_text=prompt,
            text_attention_mask=prompt_mask,
            apply_dac=apply_dac,
        )
        # hs: [num_layers, N_q, B, D] → transpose to [num_layers, B, N_q, D]
        hs = hs.transpose(1, 2)
        reference_boxes = reference_boxes.transpose(1, 2)

        # Handle DAC: only use o2o queries
        num_o2o = (hs.size(2) // 2) if apply_dac else hs.size(2)

        # ── Phase 10: DotProductScoring (fine-tuned) ─────────────────
        pred_logits = self.dot_prod_scoring(
            hs[:, :, :num_o2o], prompt, prompt_mask,
        )  # [num_layers, B, N_q, 1]

        # ── Phase 11: Segmentation Head (fine-tuned) ─────────────────
        # img_ids: assume all same image (batch dim)
        img_ids = torch.zeros(B, dtype=torch.long, device=image.device)
        seg_out = self.segmentation_head(
            backbone_feats=backbone_out["backbone_fpn"],
            obj_queries=hs[:, :, :num_o2o] if not self.segmentation_head.aux_masks else hs[:, :, :num_o2o],
            image_ids=img_ids,
            encoder_hidden_states=memory_out["memory"],
            prompt=prompt,
            prompt_mask=prompt_mask,
        )
        all_pred_masks = seg_out["pred_masks"]  # [B, N_q, H_mask, W_mask]

        # ── Phase 12: Select best mask ───────────────────────────────
        # Use last layer logits
        last_logits = pred_logits[-1]  # [B, N_q, 1]
        best_idx = last_logits.squeeze(-1).argmax(dim=1)  # [B]
        batch_idx = torch.arange(B, device=image.device)

        selected_mask = all_pred_masks[batch_idx, best_idx].unsqueeze(1)  # [B, 1, H, W]
        selected_score = last_logits[batch_idx, best_idx]  # [B, 1]

        # ── Build output ─────────────────────────────────────────────
        from sam3.model.box_ops import box_cxcywh_to_xyxy

        last_ref = reference_boxes[-1, :, :num_o2o]  # [B, N_q, 4]
        pred_boxes_xyxy = box_cxcywh_to_xyxy(last_ref)

        output: Dict[str, Tensor] = {
            "pred_masks": selected_mask,              # [B, 1, H, W]
            "pred_logits": last_logits,               # [B, N_q, 1]
            "pred_boxes": pred_boxes_xyxy,            # [B, N_q, 4]
            "selected_score": selected_score,         # [B, 1]
        }
        if return_gate_weights:
            output["gate_weights"] = gate_weights
        if return_all_queries:
            output["all_pred_masks"] = all_pred_masks

        # Cast to float32 — prevents float16 overflow (from untrained PIM
        # tokens causing large attention scores in transformer) from
        # propagating NaN to loss/eval.  AMP backward handles the cast.
        return {
            k: v.float() if isinstance(v, torch.Tensor) else v
            for k, v in output.items()
        }

    # ── K-perturbation for consistency training ──────────────────────

    def forward_k_perturbations(
        self,
        image: Tensor,
        K: int = 3,
        **prompt_kwargs,
    ) -> List[Dict[str, Tensor]]:
        """Run K independent perturbations for consistency loss."""
        return [self.forward(image, **prompt_kwargs) for _ in range(K)]

    # ── Helpers ──────────────────────────────────────────────────────

    def _get_img_feats(
        self, backbone_out: Dict[str, Tensor],
    ) -> Tuple[List[Tensor], List[Tensor], List[Tuple[int, int]]]:
        """Extract image features in the format encoder expects.

        Returns:
            img_feats:      List of [HW, B, D] (seq-first)
            img_pos_embeds: List of [HW, B, D] (seq-first)
            vis_feat_sizes: List of (H, W)
        """
        num_levels = 1  # SAM3 uses 1 feature level

        vis_feats = backbone_out["backbone_fpn"][-num_levels:]
        vis_pos_enc = backbone_out["vision_pos_enc"][-num_levels:]
        vis_feat_sizes = [x.shape[-2:] for x in vis_pos_enc]  # [(H, W)]

        # NxCxHxW => HWxNxC (batch-first => seq-first)
        img_feats = [x.flatten(2).permute(2, 0, 1) for x in vis_feats]
        img_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vis_pos_enc]

        return img_feats, img_pos_embeds, vis_feat_sizes

    def _prepare_mask_prompt(
        self, masks: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Resize mask to PromptEncoder's expected input size."""
        if masks is None:
            return None
        pe = self.geometry_encoder.prompt_encoder
        if not hasattr(pe, "mask_input_size"):
            return masks

        target = tuple(pe.mask_input_size)
        if tuple(masks.shape[-2:]) == target:
            return masks

        return F.interpolate(
            masks.float(), size=target,
            mode="bilinear", align_corners=False, antialias=True,
        )



    # ── Weight loading ───────────────────────────────────────────────

    def load_weights(
        self,
        checkpoint_path: str,
        *,
        strict: bool = False,
        map_location: str = "cpu",
    ) -> Dict:
        """Load trained UMPAv2 weights."""
        raw = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        metadata = {}
        if isinstance(raw, dict):
            for k in ("epoch", "val_dice", "val_miou"):
                if k in raw:
                    metadata[k] = raw[k]

        state_dict = raw
        if isinstance(raw, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in raw and isinstance(raw[key], dict):
                    state_dict = raw[key]
                    break

        # Strip DDP prefix
        cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}

        result = self.load_state_dict(cleaned, strict=strict)
        logger.info(
            "Loaded UMPAv2 checkpoint | epoch=%s | matched=%d missing=%d unexpected=%d",
            metadata.get("epoch", "?"),
            len(cleaned) - len(result.unexpected_keys or []),
            len(result.missing_keys or []),
            len(result.unexpected_keys or []),
        )
        return {"metadata": metadata, **result._asdict()}
