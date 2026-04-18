"""Model wrappers for embedding-fusion experiments.

Two variants that change HOW embeddings are concatenated before the Mask Decoder,
without modifying UMPAModel source code.

Wrappers
--------
- ``EFusedOnlyWrapper``:          sparse_prompt_embs = E_fused.unsqueeze(1)
- ``ESparseEFusedETextWrapper``:  sparse_prompt_embs = cat([E_sparse, E_fused, E_text])

Standard UMPAModel:               sparse_prompt_embs = cat([E_sparse, E_fused])
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from umpt_sam.umpa_model import UMPAModel


# ======================================================================
# Base mixin: delegate common methods to self.model
# ======================================================================

class _ModelWrapperBase(nn.Module):
    """Shared delegate methods for all experiment wrappers."""

    model: UMPAModel

    def forward_k_perturbations(self, image, K=3, **kwargs):
        return [self.forward(image, **kwargs) for _ in range(K)]

    def train(self, mode=True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def parameters(self, recurse=True):
        return self.model.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True):
        return self.model.named_parameters(prefix, recurse)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)


# ======================================================================
# Shared forward logic (up to UPFE output)
# ======================================================================

def _shared_forward_before_concat(model, image, boxes, points, point_labels,
                                   masks, captions, text_embeddings,
                                   return_prompt_weights):
    """Run perturbation → image encoder → text encoder → prompt encoder → UPFE.

    Returns all intermediate tensors needed to build sparse_prompt_embeddings.
    """
    single_caption = captions[0] if captions and len(captions) == 1 else None
    perturbed = model.perturbation(
        bbox=boxes, points=points, point_labels=point_labels,
        mask=masks, text=single_caption, text_embeddings=text_embeddings,
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

    # Image encoder
    no_grad = not any(p.requires_grad for p in model.image_encoder.parameters())
    with torch.set_grad_enabled(not no_grad):
        backbone_out = model.image_encoder.forward_image(image)
    image_embeddings, high_res_features = model._select_sam_backbone_outputs(backbone_out)

    if high_res_features is not None and getattr(model.sam_mask_decoder, "use_high_res_features", False):
        conv_s0 = getattr(model.sam_mask_decoder, "conv_s0", None)
        conv_s1 = getattr(model.sam_mask_decoder, "conv_s1", None)
        if conv_s0 is not None and conv_s1 is not None:
            high_res_features = [conv_s0(high_res_features[0]), conv_s1(high_res_features[1])]

    image_pe = model.prompt_encoder.get_dense_pe()

    # Text encoder
    if captions_p is not None:
        text_out = model.image_encoder.forward_text(captions_p, device=image.device)
        text_emb_p = text_out["language_features"].permute(1, 0, 2)
    text_emb_p = model._project_text_embeddings(text_emb_p)

    # Prompt encoder
    point_input = (points_p, point_labels_p) if points_p is not None else None
    sparse_embs, dense_embs = model.prompt_encoder(
        points=point_input, boxes=boxes_p, masks=masks_p,
    )

    # UPFE fusion
    upfe_input = {
        "sparse_embeddings": sparse_embs,
        "mask_embeddings": dense_embs.flatten(2).permute(0, 2, 1),
        "text_embeddings": text_emb_p,
    }
    upfe_out = model.upfe_encoder(upfe_input, return_weights=return_prompt_weights)
    if return_prompt_weights:
        e_fused, prompt_weights = upfe_out
    else:
        e_fused = upfe_out
        prompt_weights = None

    return {
        "sparse_embs": sparse_embs,
        "dense_embs": dense_embs,
        "e_fused": e_fused,
        "text_emb_p": text_emb_p,
        "prompt_weights": prompt_weights,
        "image_embeddings": image_embeddings,
        "image_pe": image_pe,
        "high_res_features": high_res_features,
    }


def _run_mask_decoder(model, ctx, sparse_prompt_embeddings, multimask_output,
                      return_prompt_weights):
    """Run mask decoder and build output dict."""
    decoder_out = model.sam_mask_decoder(
        image_embeddings=ctx["image_embeddings"],
        image_pe=ctx["image_pe"],
        sparse_prompt_embeddings=sparse_prompt_embeddings,
        dense_prompt_embeddings=ctx["dense_embs"],
        multimask_output=multimask_output,
        repeat_image=False,
        high_res_features=ctx["high_res_features"],
    )
    pred_masks, iou_predictions = decoder_out[:2]

    output = {"pred_masks": pred_masks, "iou_predictions": iou_predictions}
    if return_prompt_weights:
        output["prompt_weights"] = ctx["prompt_weights"]
    return output


# ======================================================================
# Wrapper 1: E_fused only
# ======================================================================

class EFusedOnlyWrapper(_ModelWrapperBase):
    """Mask Decoder receives ONLY E_fused (no E_sparse).

    Standard:   cat([E_sparse, E_fused.unsqueeze(1)])
    This:       E_fused.unsqueeze(1)
    """

    def __init__(self, base_model: UMPAModel):
        super().__init__()
        self.model = base_model

    def forward(self, image, boxes=None, points=None, point_labels=None,
                masks=None, captions=None, text_embeddings=None,
                multimask_output=False, return_prompt_weights=False,
                **kwargs) -> Dict[str, torch.Tensor]:

        ctx = _shared_forward_before_concat(
            self.model, image, boxes, points, point_labels,
            masks, captions, text_embeddings, return_prompt_weights,
        )

        # ============ KEY: Only E_fused ============
        sparse_prompt_embeddings = ctx["e_fused"].unsqueeze(1)

        return _run_mask_decoder(
            self.model, ctx, sparse_prompt_embeddings,
            multimask_output, return_prompt_weights,
        )


# ======================================================================
# Wrapper 2: E_sparse + E_fused + E_text
# ======================================================================

class ESparseEFusedETextWrapper(_ModelWrapperBase):
    """Mask Decoder receives E_sparse + E_fused + E_text.

    Standard:   cat([E_sparse, E_fused.unsqueeze(1)])
    This:       cat([E_sparse, E_fused.unsqueeze(1), E_text])
    """

    def __init__(self, base_model: UMPAModel):
        super().__init__()
        self.model = base_model

    def forward(self, image, boxes=None, points=None, point_labels=None,
                masks=None, captions=None, text_embeddings=None,
                multimask_output=False, return_prompt_weights=False,
                **kwargs) -> Dict[str, torch.Tensor]:

        ctx = _shared_forward_before_concat(
            self.model, image, boxes, points, point_labels,
            masks, captions, text_embeddings, return_prompt_weights,
        )

        # ============ KEY: E_sparse + E_fused + E_text ============
        parts = [ctx["sparse_embs"], ctx["e_fused"].unsqueeze(1)]
        if ctx["text_emb_p"] is not None:
            parts.append(ctx["text_emb_p"])
        sparse_prompt_embeddings = torch.cat(parts, dim=1)

        return _run_mask_decoder(
            self.model, ctx, sparse_prompt_embeddings,
            multimask_output, return_prompt_weights,
        )
