from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

# Relative imports work when this module is imported as part of the `sam3` package
# (e.g., `python -m sam3.umpt_sam.umpa_model`). When running as a script, fall back
# to absolute imports so the module still works.
import sys

sys.path.append("sam3")  # Ensure current directory is in path for absolute imports
try:
    from ..sam3.model.vl_combiner import SAM3VLBackbone
    from ..sam3.sam.mask_decoder import MaskDecoder
    from ..sam3.sam.prompt_encoder import PromptEncoder
except ImportError:  # pragma: no cover
    from sam3.model.vl_combiner import SAM3VLBackbone
    from sam3.sam.mask_decoder import MaskDecoder
    from sam3.sam.prompt_encoder import PromptEncoder

from .config.model_config import UMPAModelConfig, GRCMIConfig
from .modules.modules import PromptPerturbation
from .modules.upf_enconder import UnifiedPromptFusionEncoder
from .modules.grcmi import GatedResidualInjector


class UMPAModel(nn.Module):
    """Unified Multi-Prompt Adaptation model built on top of SAM3."""

    @classmethod
    def from_config(
        cls,
        model_config: UMPAModelConfig,
        image_encoder: Optional[SAM3VLBackbone] = None,
        prompt_encoder: Optional[PromptEncoder] = None,
        mask_decoder: Optional[MaskDecoder] = None,
        perturbation_cfg: Optional[dict] = None,
        sam_build_kwargs: Optional[dict] = None,
        checkpoint_path: Optional[str] = None,
        map_location: str = "cpu",
    ) -> "UMPAModel":
        """Build UMPAModel from config and optionally load trained weights.

        When ``checkpoint_path`` is provided the SAM3 architecture is built
        **without** loading the original SAM3 weights (lightweight), then
        *all* weights are loaded directly from the trained checkpoint.

        When ``checkpoint_path`` is ``None`` the original SAM3 checkpoint is
        loaded via :meth:`load_sam_components` so the model is ready for
        training / fine-tuning.

        Parameters
        ----------
        checkpoint_path : str | None
            Path to trained UMPA ``.pth`` checkpoint.  If given, SAM3
            weights are **not** loaded separately.
        map_location : str
            Device for weight loading, forwarded to :func:`torch.load`.
        """
        # Resolve checkpoint_path: explicit arg > model_config field
        if checkpoint_path is None:
            checkpoint_path = getattr(model_config, "checkpoint_path", None)

        if checkpoint_path is not None:
            # ---- Lightweight path: architecture only, no SAM3 weights ----
            if image_encoder is None or prompt_encoder is None or mask_decoder is None:
                image_encoder, prompt_encoder, mask_decoder = (
                    cls._build_sam_architecture(**(sam_build_kwargs or {}))
                )
        else:
            # ---- Training path: load SAM3 pre-trained weights ------------
            if image_encoder is None or prompt_encoder is None or mask_decoder is None:
                image_encoder, prompt_encoder, mask_decoder = cls.load_sam_components(
                    model_config=model_config,
                    **(sam_build_kwargs or {}),
                )

        model = cls(
            image_encoder=image_encoder,
            prompt_encoder=prompt_encoder,
            mask_decoder=mask_decoder,
            embed_dim=model_config.embed_dim,
            text_embed_dim=model_config.text_embed_dim,
            freeze_image_encoder=model_config.freeze_image_encoder,
            upfe_hidden_dim=model_config.upfe.scoring_hidden_dim,
            perturbation_cfg=None,
        )
        model.upfe_encoder = UnifiedPromptFusionEncoder.from_config(model_config.upfe)
        model.grcmi = GatedResidualInjector.from_config(model_config.grcmi)
        model.perturbation = PromptPerturbation.from_config(
            model_config.mppg,
            **(perturbation_cfg or {}),
        )
        model.model_config = model_config

        if checkpoint_path is not None:
            model.load_weights(
                checkpoint_path,
                skip_image_encoder=False,   # load ALL weights from checkpoint
                map_location=map_location,
            )

        return model

    @classmethod
    def _build_sam_architecture(
        cls,
        **kwargs,
    ) -> Tuple[SAM3VLBackbone, PromptEncoder, MaskDecoder]:
        """Build SAM3 module architecture **without** loading any weights.

        This is used when we already have a trained checkpoint that
        contains all weights — avoids the expensive SAM3 checkpoint load.
        """
        from sam3.model_builder import (
            _create_vision_backbone,
            _create_text_encoder,
            _create_vl_backbone,
            build_tracker,
        )
        import os

        bpe_path = kwargs.get("bpe_path")
        if bpe_path is None:
            import sam3
            bpe_path = os.path.join(
                os.path.dirname(sam3.__file__),
                "..", "assets", "bpe_simple_vocab_16e6.txt.gz",
            )

        vision_encoder = _create_vision_backbone(enable_inst_interactivity=True)
        text_encoder = _create_text_encoder(bpe_path)
        image_encoder = _create_vl_backbone(vision_encoder, text_encoder)

        tracker = build_tracker(apply_temporal_disambiguation=False)
        prompt_encoder = tracker.sam_prompt_encoder
        mask_decoder = tracker.sam_mask_decoder

        return image_encoder, prompt_encoder, mask_decoder

    @classmethod
    def load_sam_components(
        cls,
        model_config: UMPAModelConfig,
        device: str = "cpu",
        eval_mode: bool = False,
        load_from_HF: bool = False,
        enable_inst_interactivity: bool = True,
    ) -> Tuple[SAM3VLBackbone, PromptEncoder, MaskDecoder]:
        """Load SAM3 backbone, prompt encoder, and mask decoder from checkpoint."""
        from sam3 import build_sam3_image_model

        sam_model = build_sam3_image_model(
            checkpoint_path=str(model_config.sam_checkpoint_path),
            device=device,
            eval_mode=eval_mode,
            load_from_HF=load_from_HF,
            enable_inst_interactivity=enable_inst_interactivity,
        )
        image_encoder = sam_model.backbone

        interactive_predictor = getattr(sam_model, "inst_interactive_predictor", None)
        tracker_model = getattr(interactive_predictor, "model", None)
        if tracker_model is None:
            raise RuntimeError(
                "SAM interactive heads are unavailable. "
                "Provide prompt_encoder/mask_decoder explicitly or load a checkpoint with tracker heads."
            )

        return (
            image_encoder,
            tracker_model.sam_prompt_encoder,
            tracker_model.sam_mask_decoder,
        )
    
    def __init__(
        self,
        image_encoder: SAM3VLBackbone,
        prompt_encoder: PromptEncoder,
        mask_decoder: MaskDecoder,
        embed_dim: int = 256,
        text_embed_dim: int = 512,
        freeze_image_encoder: bool = True,
        upfe_hidden_dim: int = 256,
        perturbation_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.model_config: Optional[UMPAModelConfig] = None

        self.image_encoder = image_encoder
        if freeze_image_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

        self.prompt_encoder = prompt_encoder
        self.sam_mask_decoder = mask_decoder

        self.embed_dim = embed_dim
        self.text_embed_dim = text_embed_dim
        self.text_projection: nn.Module
        if text_embed_dim == embed_dim:
            self.text_projection = nn.Identity()
        else:
            self.text_projection = nn.Linear(text_embed_dim, embed_dim)

        self.upfe_encoder = UnifiedPromptFusionEncoder(
            embed_dim=embed_dim,
            scoting_network_hidden_dim=upfe_hidden_dim,
        )
        self.grcmi = GatedResidualInjector(embed_dim=embed_dim)
        self.perturbation = PromptPerturbation(**(perturbation_cfg or {}))

    @staticmethod
    def _select_sam_backbone_outputs(
        backbone_out: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        sam_backbone_out = backbone_out.get("sam2_backbone_out") or backbone_out
        image_embeddings = sam_backbone_out["vision_features"]

        high_res_features = None
        backbone_fpn = sam_backbone_out.get("backbone_fpn")
        if backbone_fpn is not None and len(backbone_fpn) >= 2:
            high_res_features = [backbone_fpn[0], backbone_fpn[1]]

        return image_embeddings, high_res_features

    def _prepare_mask_prompt(
        self, masks: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if masks is None or not hasattr(self.prompt_encoder, "mask_input_size"):
            return masks

        target_size = tuple(self.prompt_encoder.mask_input_size)
        if tuple(masks.shape[-2:]) == target_size:
            return masks

        return F.interpolate(
            masks.float(),
            size=target_size,
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def _project_text_embeddings(
        self,
        text_embeddings: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if text_embeddings is None:
            return None

        if text_embeddings.dim() == 2:
            text_embeddings = text_embeddings.unsqueeze(1)

        if text_embeddings.shape[-1] == self.embed_dim:
            return text_embeddings

        if text_embeddings.shape[-1] != self.text_embed_dim:
            raise ValueError(
                f"Expected text embedding dim {self.embed_dim} or {self.text_embed_dim}, "
                f"got {text_embeddings.shape[-1]}."
            )

        return self.text_projection(text_embeddings)

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
        """Run one UMPA forward pass."""
        single_caption = captions[0] if captions and len(captions) == 1 else None
        perturbed = self.perturbation(
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
        masks_p = self._prepare_mask_prompt(perturbed.get("mask", masks))
        text_emb_p = perturbed.get("text_embedding", text_embeddings)

        if captions is None:
            captions_p = None
        elif len(captions) == 1 and "text" in perturbed:
            captions_p = [perturbed["text"]]
        else:
            captions_p = captions

        no_grad = not any(
            param.requires_grad for param in self.image_encoder.parameters()
        )
        with torch.set_grad_enabled(not no_grad):
            backbone_out = self.image_encoder.forward_image(image)
        image_embeddings, high_res_features = self._select_sam_backbone_outputs(
            backbone_out
        )

        if high_res_features is not None and getattr(
            self.sam_mask_decoder, "use_high_res_features", False
        ):
            conv_s0 = getattr(self.sam_mask_decoder, "conv_s0", None)
            conv_s1 = getattr(self.sam_mask_decoder, "conv_s1", None)
            if conv_s0 is not None and conv_s1 is not None:
                high_res_features = [
                    conv_s0(high_res_features[0]),
                    conv_s1(high_res_features[1]),
                ]

        image_pe = self.prompt_encoder.get_dense_pe()

        if captions_p is not None:
            text_out = self.image_encoder.forward_text(captions_p, device=image.device)
            text_emb_p = text_out["language_features"].permute(1, 0, 2)
        text_emb_p = self._project_text_embeddings(text_emb_p)
        point_input = (points_p, point_labels_p) if points_p is not None else None
        sparse_embs, dense_embs = self.prompt_encoder(
            points=point_input,
            boxes=boxes_p,
            masks=masks_p,
        )

        upfe_input: Dict[str, Optional[torch.Tensor]] = {
            "sparse_embeddings": sparse_embs,
            "mask_embeddings": dense_embs.flatten(2).permute(0, 2, 1),
            "text_embeddings": text_emb_p,
        }
        upfe_out = self.upfe_encoder(upfe_input, return_weights=return_prompt_weights)
        if return_prompt_weights:
            e_fused, prompt_weights = upfe_out
        else:
            e_fused = upfe_out
            prompt_weights = None

        sparse_prompt_embeddings = torch.cat([sparse_embs, e_fused.unsqueeze(1)], dim=1)
        decoder_out = self.sam_mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
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
            output["prompt_weights"] = prompt_weights
        return output

    def forward_k_perturbations(
        self,
        image: torch.Tensor,
        K: int = 3,
        **prompt_kwargs,
    ) -> List[Dict[str, torch.Tensor]]:
        """Run K independent prompt perturbations for consistency training."""
        return [self.forward(image, **prompt_kwargs) for _ in range(K)]

    def freeze_layers(self, layers: List[str]) -> torch.Tensor:
        """Freeze specified layers by name."""
        for layer in layers:
            module = getattr(self, layer, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"Layer {layer} not found in model.")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        checkpoint_path: str,
        *,
        strict: bool = False,
        skip_image_encoder: bool = True,
        map_location: str = "cpu",
    ) -> Dict:
        """Load trained UMPA weights from a checkpoint file.

        The expected checkpoint format (as produced by the training loop)::

            {
                "epoch": int,
                "model_state_dict": OrderedDict,
                "optimizer_state_dict": dict,
                "val_dice": float,
                "val_miou": float,
            }

        Parameters
        ----------
        checkpoint_path : str
            Path to the ``.pt`` / ``.pth`` checkpoint produced by training.
        strict : bool, default False
            If ``True``, require an exact match between checkpoint keys and
            model keys (after optional filtering).  ``False`` allows partial
            loading — useful when the checkpoint contains all layers but
            ``skip_image_encoder`` drops the backbone keys.
        skip_image_encoder : bool, default True
            If ``True``, keys prefixed with ``image_encoder.`` are removed
            from the checkpoint *before* loading.  This prevents
            overwriting the frozen SAM backbone already initialised by
            :meth:`load_sam_components`.
        map_location : str, default "cpu"
            Device string forwarded to :func:`torch.load`.

        Returns
        -------
        dict
            ``{"matched", "missing", "unexpected"}`` key lists **plus**
            training metadata: ``"epoch"``, ``"val_dice"``, ``"val_miou"``.
        """
        import logging

        logger = logging.getLogger(__name__)

        # --- 1. Load checkpoint ------------------------------------------
        raw = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False,
        )

        # --- 2. Extract training metadata --------------------------------
        metadata: Dict = {}
        if isinstance(raw, dict):
            for meta_key in ("epoch", "val_dice", "val_miou"):
                if meta_key in raw:
                    metadata[meta_key] = raw[meta_key]

        # --- 3. Extract model state dict ---------------------------------
        state_dict = raw
        if isinstance(raw, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in raw and isinstance(raw[key], dict):
                    state_dict = raw[key]
                    break

        # --- 4. Strip DDP "module." prefix if present --------------------
        cleaned: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            cleaned[k.removeprefix("module.")] = v

        # --- 5. Optionally drop frozen backbone keys ---------------------
        if skip_image_encoder:
            before = len(cleaned)
            cleaned = {
                k: v
                for k, v in cleaned.items()
                if not k.startswith("image_encoder.")
            }
            skipped = before - len(cleaned)
            if skipped:
                logger.info("Skipped %d image_encoder keys.", skipped)

        # --- 6. Load into model ------------------------------------------
        result = self.load_state_dict(cleaned, strict=strict)

        matched = [
            k for k in cleaned if k not in (result.unexpected_keys or [])
        ]
        info: Dict = {
            "matched": matched,
            "missing": list(result.missing_keys or []),
            "unexpected": list(result.unexpected_keys or []),
            **metadata,
        }

        logger.info(
            "Checkpoint loaded  |  epoch=%s  val_dice=%.4f  val_miou=%.4f",
            metadata.get("epoch", "?"),
            metadata.get("val_dice", 0.0),
            metadata.get("val_miou", 0.0),
        )
        logger.info(
            "Keys  |  matched=%d  missing=%d  unexpected=%d",
            len(matched),
            len(info["missing"]),
            len(info["unexpected"]),
        )
        if info["missing"]:
            logger.warning("Missing keys: %s", info["missing"])
        if info["unexpected"]:
            logger.warning("Unexpected keys: %s", info["unexpected"])

        return info

    @classmethod
    def from_pretrained(
        cls,
        model_config: UMPAModelConfig,
        checkpoint_path: str,
        *,
        strict: bool = False,
        skip_image_encoder: bool = True,
        map_location: str = "cpu",
        sam_build_kwargs: Optional[dict] = None,
    ) -> "UMPAModel":
        """Build UMPAModel from config and load trained weights in one call.

        Parameters
        ----------
        model_config : UMPAModelConfig
            Full model configuration (SAM checkpoint, embed dims, etc.).
        checkpoint_path : str
            Path to the trained UMPA checkpoint.
        strict, skip_image_encoder, map_location
            Forwarded to :meth:`load_weights`.
        sam_build_kwargs : dict | None
            Extra kwargs forwarded to :meth:`load_sam_components`.

        Returns
        -------
        UMPAModel
            Fully constructed model with trained weights loaded.
        """
        model = cls.from_config(model_config, sam_build_kwargs=sam_build_kwargs)
        model.load_weights(
            checkpoint_path,
            strict=strict,
            skip_image_encoder=skip_image_encoder,
            map_location=map_location,
        )
        return model

if __name__ == "__main__":
    # Example usage
    from umpt_sam.config.model_config import UPFEConfig, MPPGConfig

    model_config = UMPAModelConfig(
        sam_checkpoint="sam3.pt",
        embed_dim=256,
        text_embed_dim=512,
        freeze_image_encoder=True,
        upfe=UPFEConfig(scoring_hidden_dim=256),
        mppg=MPPGConfig(),
    )
    print("Model config:", model_config)
    print("Building model...")
    model = UMPAModel.from_config(model_config=model_config)
    
    print("Model built.")
    dummy_image = cv2.imread("mask.png")  # Replace with actual image path
    dummy_image = cv2.resize(dummy_image, (1008, 1008))
    dummy_image = torch.from_numpy(dummy_image).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    dummy_boxes = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float)
    dummy_points = torch.tensor([[[504, 504]]], dtype=torch.float)
    dummy_point_labels = torch.tensor([[1]])
    dummy_captions = ["polyp"]
    output = model(
        image=dummy_image,
        boxes=dummy_boxes,
        points=dummy_points,
        point_labels=dummy_point_labels,
        captions=dummy_captions,
    )
    cv2.imwrite("pred_mask.png", (output['pred_masks'][0, 0].cpu().detach().numpy() * 255).astype('uint8'))
    print(output['pred_masks'].shape, output['iou_predictions'])
