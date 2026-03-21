"""Ablation model wrapper for UMPT-SAM.

Wraps ``UMPAModel`` to selectively disable prompts and components
for ablation studies. The original ``UMPAModel`` is NOT modified.

Usage
-----
>>> from umpt_sam.ablation_wrapper import AblationModelWrapper
>>> from umpt_sam.config.experiment_config import get_scenario
>>> wrapper = AblationModelWrapper.wrap(base_model, get_scenario("only_box"))
>>> output = wrapper(image=img, boxes=boxes, points=pts, ...)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config.experiment_config import ExperimentConfig
from .umpa_model import UMPAModel


class AblationModelWrapper(nn.Module):
    """Wrapper that intercepts ``UMPAModel.forward()`` to apply ablation
    controls WITHOUT modifying the original model code.

    Three control mechanisms:
    1. **Prompt filtering**: Set unused prompt args to ``None``.
    2. **MPPG bypass**: Force ``perturbation.eval()`` → identity.
    3. **UPFE bypass**: Skip fusion, use raw ``sparse_embs`` only.
    """

    def __init__(self, base_model: UMPAModel, exp_cfg: ExperimentConfig):
        super().__init__()
        # Share all weights — do NOT copy or re-init
        self.model = base_model
        self.exp_cfg = exp_cfg

    @classmethod
    def wrap(
        cls,
        base_model: UMPAModel,
        exp_cfg: ExperimentConfig,
    ) -> "AblationModelWrapper":
        """Factory method to wrap an existing model."""
        return cls(base_model, exp_cfg)

    # ------------------------------------------------------------------
    # Prompt filtering
    # ------------------------------------------------------------------

    def _filter_prompts(
        self,
        boxes: Optional[torch.Tensor],
        points: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        captions: Optional[List[str]],
        text_embeddings: Optional[torch.Tensor],
    ) -> dict:
        """Zero-out disabled prompts according to experiment config."""
        if not self.exp_cfg.use_box:
            boxes = None
        if not self.exp_cfg.use_point:
            points = None
            point_labels = None
        if not self.exp_cfg.use_mask:
            masks = None
        if not self.exp_cfg.use_text:
            captions = None
            text_embeddings = None
        return {
            "boxes": boxes,
            "points": points,
            "point_labels": point_labels,
            "masks": masks,
            "captions": captions,
            "text_embeddings": text_embeddings,
        }

    # ------------------------------------------------------------------
    # Forward — with UPFE
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_prompt_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with ablation controls applied."""

        # 1. Filter prompts
        filtered = self._filter_prompts(
            boxes, points, point_labels, masks, captions, text_embeddings,
        )

        # 2. MPPG control
        mppg_was_training = self.model.perturbation.training
        if not self.exp_cfg.enable_mppg:
            self.model.perturbation.eval()

        try:
            # 3. Route to appropriate forward
            if self.exp_cfg.enable_upfe:
                result = self.model.forward(
                    image=image,
                    multimask_output=multimask_output,
                    return_prompt_weights=return_prompt_weights,
                    **filtered,
                )
            else:
                result = self._forward_without_upfe(
                    image=image,
                    multimask_output=multimask_output,
                    return_prompt_weights=return_prompt_weights,
                    **filtered,
                )
        finally:
            # 4. Restore MPPG state
            if not self.exp_cfg.enable_mppg and mppg_was_training:
                self.model.perturbation.train()

        return result

    # ------------------------------------------------------------------
    # Forward WITHOUT UPFE — duplicated logic to avoid modifying original
    # ------------------------------------------------------------------

    def _forward_without_upfe(
        self,
        image: torch.Tensor,
        boxes: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        captions: Optional[List[str]] = None,
        text_embeddings: Optional[torch.Tensor] = None,
        multimask_output: bool = False,
        return_prompt_weights: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """UMPAModel.forward() logic with UPFE bypassed.

        Instead of:
            sparse_prompt_embeddings = cat([sparse_embs, e_fused])
        We use:
            sparse_prompt_embeddings = sparse_embs  (raw, no fusion)

        This is intentional code duplication to preserve the original
        UMPAModel source untouched.
        """
        model = self.model

        # --- Perturbation ---
        single_caption = captions[0] if captions and len(captions) == 1 else None
        perturbed = model.perturbation(
            bbox=boxes,
            points=points,
            point_labels=point_labels,
            mask=masks,
            text=single_caption,
            text_embeddings=text_embeddings,
        )

        boxes_p = perturbed.get("bbox", boxes)
        points_p = perturbed.get("points", points)
        point_labels_p = perturbed.get("point_labels", point_labels)
        masks_p = model._prepare_mask_prompt(perturbed.get("mask", masks))
        text_emb_p = perturbed.get("text_embedding", text_embeddings)

        if captions is None:
            captions_p = None
        elif len(captions) == 1 and "text" in perturbed:
            captions_p = [perturbed["text"]]
        else:
            captions_p = captions

        # --- Image encoder ---
        no_grad = not any(
            param.requires_grad for param in model.image_encoder.parameters()
        )
        with torch.set_grad_enabled(not no_grad):
            backbone_out = model.image_encoder.forward_image(image)
        image_embeddings, high_res_features = model._select_sam_backbone_outputs(
            backbone_out
        )

        if high_res_features is not None and getattr(
            model.sam_mask_decoder, "use_high_res_features", False
        ):
            conv_s0 = getattr(model.sam_mask_decoder, "conv_s0", None)
            conv_s1 = getattr(model.sam_mask_decoder, "conv_s1", None)
            if conv_s0 is not None and conv_s1 is not None:
                high_res_features = [
                    conv_s0(high_res_features[0]),
                    conv_s1(high_res_features[1]),
                ]

        image_pe = model.prompt_encoder.get_dense_pe()

        # --- Text encoder ---
        if captions_p is not None:
            text_out = model.image_encoder.forward_text(captions_p, device=image.device)
            text_emb_p = text_out["language_features"].permute(1, 0, 2)
        text_emb_p = model._project_text_embeddings(text_emb_p)

        # --- Prompt encoder ---
        point_input = (points_p, point_labels_p) if points_p is not None else None
        sparse_embs, dense_embs = model.prompt_encoder(
            points=point_input,
            boxes=boxes_p,
            masks=masks_p,
        )

        # === KEY DIFFERENCE: Skip UPFE, use sparse_embs directly ===
        sparse_prompt_embeddings = sparse_embs

        # --- Mask decoder ---
        decoder_out = model.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_embs,
            multimask_output=multimask_output,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        pred_masks, iou_predictions = decoder_out[:2]

        output: Dict[str, torch.Tensor] = {
            "pred_masks": pred_masks,
            "iou_predictions": iou_predictions,
        }
        if return_prompt_weights:
            output["prompt_weights"] = None  # No UPFE → no weights
        return output

    # ------------------------------------------------------------------
    # Delegate remaining methods to base model
    # ------------------------------------------------------------------

    def forward_k_perturbations(
        self,
        image: torch.Tensor,
        K: int = 3,
        **prompt_kwargs,
    ) -> list:
        """Run K perturbation forward passes through the wrapper."""
        return [self.forward(image, **prompt_kwargs) for _ in range(K)]

    def train(self, mode: bool = True):
        """Propagate train mode to base model."""
        self.model.train(mode)
        return self

    def eval(self):
        """Propagate eval mode to base model."""
        self.model.eval()
        return self

    def parameters(self, recurse: bool = True):
        return self.model.parameters(recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self.model.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
