import torch
import torch.nn as nn
import math
from typing import Tuple, Optional

from ..config.model_config import MPPGConfig


class BBoxPerturbation(nn.Module):
    """
    Bounding Box Perturbation
    
    Given a ground-truth bounding box B = (x1, y1, x2, y2), generates:
        B̃ = (x1 + δ1, y1 + δ2, x2 + δ3, y2 + δ4)
    
    where δi ~ N(0, σ²_B) + γi for asymmetric errors,
    and optionally applies a mild rotation θ ~ U(-3°, 3°).
    """
    
    def __init__(
        self,
        sigma_b: float = 5.0,           
        gamma_range: Tuple[float, float] = (-2.0, 2.0),  
        rotation_range: float = 3.0, 
        apply_rotation: bool = True,
    ):
        super().__init__()
        self.sigma_b = sigma_b
        self.gamma_range = gamma_range
        self.rotation_range = rotation_range
        self.apply_rotation = apply_rotation
    
    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: (B, N, 4) or (B, 4) - bounding boxes as (x1, y1, x2, y2)
        
        Returns:
            Perturbed bounding boxes with same shape
        """
        if not self.training:
            return bbox
        
        original_shape = bbox.shape
        if bbox.dim() == 2:
            bbox = bbox.unsqueeze(1)  # (B, 4) -> (B, 1, 4)
        
        B, N, _ = bbox.shape
        device = bbox.device
        
        # δi ~ N(0, σ²_B)
        delta = torch.randn(B, N, 4, device=device) * self.sigma_b
        
        # γi ~ U(gamma_range) for asymmetric errors
        gamma = torch.empty(B, N, 4, device=device).uniform_(*self.gamma_range)
        
        # Perturbed box: B̃ = B + δ + γ
        perturbed = bbox + delta + gamma
        
        # Apply rotation if enabled
        if self.apply_rotation:
            perturbed = self._apply_rotation(perturbed)
        
        # Ensure valid box (x1 < x2, y1 < y2)
        perturbed = self._ensure_valid_box(perturbed)
        
        if len(original_shape) == 2:
            perturbed = perturbed.squeeze(1)
        
        return perturbed
    
    def _apply_rotation(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Apply mild rotation θ ~ U(-rotation_range°, rotation_range°)
        Rotates box corners around center, then computes new axis-aligned bbox.
        """
        B, N, _ = bbox.shape
        device = bbox.device
        
        # Sample rotation angles
        theta = torch.empty(B, N, device=device).uniform_(
            -self.rotation_range, self.rotation_range
        )
        theta_rad = theta * (math.pi / 180.0)
        
        # Get box corners
        x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        
        # Center of box
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Four corners relative to center
        corners_x = torch.stack([x1 - cx, x2 - cx, x2 - cx, x1 - cx], dim=-1)
        corners_y = torch.stack([y1 - cy, y1 - cy, y2 - cy, y2 - cy], dim=-1)
        
        # Rotate corners
        cos_t = torch.cos(theta_rad).unsqueeze(-1)
        sin_t = torch.sin(theta_rad).unsqueeze(-1)
        
        rotated_x = corners_x * cos_t - corners_y * sin_t + cx.unsqueeze(-1)
        rotated_y = corners_x * sin_t + corners_y * cos_t + cy.unsqueeze(-1)
        
        # New axis-aligned bbox from rotated corners
        new_x1 = rotated_x.min(dim=-1).values
        new_y1 = rotated_y.min(dim=-1).values
        new_x2 = rotated_x.max(dim=-1).values
        new_y2 = rotated_y.max(dim=-1).values
        
        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)
    
    def _ensure_valid_box(self, bbox: torch.Tensor) -> torch.Tensor:
        """Ensure x1 < x2 and y1 < y2"""
        x1, y1, x2, y2 = bbox[..., 0], bbox[..., 1], bbox[..., 2], bbox[..., 3]
        new_x1 = torch.min(x1, x2)
        new_x2 = torch.max(x1, x2)
        new_y1 = torch.min(y1, y2)
        new_y2 = torch.max(y1, y2)
        return torch.stack([new_x1, new_y1, new_x2, new_y2], dim=-1)


class PointPerturbation(nn.Module):
    """
    Positive and negative point prompts are perturbed using:
        p̃ = p + ε, where ε ~ N(0, σ²_P)
    
    Occasionally flips label from positive to negative (or vice versa)
    with probability q_flip to simulate clinical mis-clicks.
    """
    
    def __init__(
        self,
        sigma_p: float = 3.0,    
        q_flip: float = 0.05,
    ):
        super().__init__()
        self.sigma_p = sigma_p
        self.q_flip = q_flip
    
    def forward(
        self,
        points: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            points: (B, N, 2) - point coordinates (x, y)
            labels: (B, N) - point labels (1=positive, 0=negative), optional
        
        Returns:
            Tuple of (perturbed_points, perturbed_labels)
        """
        if not self.training:
            return points, labels
        
        B, N, _ = points.shape
        device = points.device
        
        # ε ~ N(0, σ²_P)
        epsilon = torch.randn(B, N, 2, device=device) * self.sigma_p
        
        # p̃ = p + ε
        perturbed_points = points + epsilon
        
        # Flip labels with probability q_flip
        perturbed_labels = labels
        if labels is not None and self.q_flip > 0:
            flip_mask = torch.rand(B, N, device=device) < self.q_flip
            # Flip: 1 -> 0, 0 -> 1
            perturbed_labels = torch.where(flip_mask, 1 - labels, labels)
        
        return perturbed_points, perturbed_labels


class MaskPerturbation(nn.Module):
    """
    A coarse mask M is produced by applying random operations:
        M = Dilate(Y, r1) ∪ Erode(Y, r2) ∪ Warp(Y, w)
    where r1, r2 simulate over/under-segmentation, and
    Warp models physiological boundary deformation.
    
    IMPORTANT: Only one operation is applied per call (mutually exclusive).
    """
    def __init__(
        self,
        dilate_radius: int = 1,
        erode_radius: int = 1,
        warp_strength: float = 0.02,
        p_dilate: float = 0.2,
        p_erode: float = 0.2,
        p_warp: float = 0.2,
        min_area_ratio: float = 0.3,
    ):
        super().__init__()
        self.dilate_radius = dilate_radius
        self.erode_radius = erode_radius
        self.warp_strength = warp_strength
        self.p_dilate = p_dilate
        self.p_erode = p_erode
        self.p_warp = p_warp
        self.min_area_ratio = min_area_ratio

    def _normalize(self, mask: torch.Tensor) -> torch.Tensor:
        """Normalize mask to [0, 1] range"""
        if mask.max() > 1:
            return mask / 255.0
        return mask

    def _denormalize(self, mask: torch.Tensor, original_max: float) -> torch.Tensor:
        """Denormalize mask back to original range"""
        if original_max > 1:
            return mask * 255.0
        return mask

    def _dilate(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Binary dilation using max_pool2d"""
        input_dim = mask.dim()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        kernel_size = 2 * radius + 1
        padding = radius
        dilated = torch.nn.functional.max_pool2d(
            mask.float(), kernel_size, stride=1, padding=padding
        )
        
        if input_dim == 3:
            return dilated.squeeze(1)
        elif input_dim == 2:
            return dilated.squeeze(0).squeeze(0)
        return dilated

    def _erode(self, mask: torch.Tensor, radius: int) -> torch.Tensor:
        """Binary erosion = 1 - dilate(1 - mask)"""
        return 1.0 - self._dilate(1.0 - mask, radius)

    def _warp(self, mask: torch.Tensor, strength: float) -> torch.Tensor:
        """Gentle elastic deformation using grid_sample"""
        input_dim = mask.dim()
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        elif mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        
        B, C, H, W = mask.shape
        device = mask.device
        
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        grid = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        
        displacement = torch.randn(B, H, W, 2, device=device) * strength
        warped_grid = grid + displacement
        warped_grid = warped_grid.clamp(-1, 1)
        
        warped = torch.nn.functional.grid_sample(
            mask.float(), warped_grid, mode='bilinear', 
            padding_mode='border', align_corners=True
        )
        warped = (warped > 0.5).float()
        
        if input_dim == 3:
            return warped.squeeze(1)
        elif input_dim == 2:
            return warped.squeeze(0).squeeze(0)
        return warped

    def _check_area(self, original: torch.Tensor, perturbed: torch.Tensor) -> bool:
        """Check if perturbed mask preserves enough area"""
        orig_area = original.sum()
        if orig_area == 0:
            return True
        pert_area = perturbed.sum()
        ratio = pert_area / orig_area
        return ratio >= self.min_area_ratio

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mask
        
        original_max = mask.max().item()
        normalized = self._normalize(mask)
        result = normalized.clone()
        
        rand_val = torch.rand(1).item()
        cumsum = 0.0
        
        cumsum += self.p_dilate
        if rand_val < cumsum:
            result = self._dilate(normalized, self.dilate_radius)
        elif rand_val < cumsum + self.p_erode:
            candidate = self._erode(normalized, self.erode_radius)
            if self._check_area(normalized, candidate):
                result = candidate
        elif rand_val < cumsum + self.p_erode + self.p_warp:
            candidate = self._warp(normalized, self.warp_strength)
            if self._check_area(normalized, candidate):
                result = candidate
        
        return self._denormalize(result, original_max)


class TextPerturbation(nn.Module):
    """
    Text Prompt Perturbation (Section 4.2.4)
    
    We represent text prompts via CLIP embeddings:
        E_text = f_CLIP(s)
    
    and add embedding noise:
        Ẽ_text = E_text + η, η ~ N(0, σ²_T)
    
    as well as synonym substitution (e.g., "polyp lesion" → "colonic lesion").
    """
    
    MEDICAL_SYNONYMS = {
        "polyp": ["lesion", "growth", "mass", "nodule", "tumor"],
        "lesion": ["polyp", "abnormality", "growth", "mass"],
        "colonic": ["colon", "colorectal", "intestinal", "bowel"],
        "adenoma": ["adenomatous polyp", "precancerous polyp", "neoplasm"],
        "tumor": ["mass", "growth", "neoplasm", "carcinoma"],
        "inflammation": ["swelling", "irritation", "erythema"],
        "ulcer": ["erosion", "sore", "wound"],
        "bleeding": ["hemorrhage", "blood loss"],
        "mucosa": ["mucosal lining", "epithelium", "membrane"],
        "sessile": ["flat", "broad-based"],
        "pedunculated": ["stalked", "protruding"],
    }
    
    def __init__(
        self,
        sigma_t: float = 0.1,
        p_noise: float = 0.5,
        p_synonym: float = 0.3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.sigma_t = sigma_t
        self.p_noise = p_noise
        self.p_synonym = p_synonym
        self.embed_dim = embed_dim
    
    def add_embedding_noise(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to embedding:
            Ẽ_text = E_text + η, η ~ N(0, σ²_T)
        """
        if torch.rand(1).item() >= self.p_noise:
            return embedding
        
        noise = torch.randn_like(embedding) * self.sigma_t
        return embedding + noise
    
    def substitute_synonyms(self, text: str) -> str:
        """
        Apply synonym substitution for medical terms.
        """
        if torch.rand(1).item() >= self.p_synonym:
            return text
        
        import random
        words = text.lower().split()
        result = []
        
        for word in words:
            clean_word = word.strip(".,;:!?")
            if clean_word in self.MEDICAL_SYNONYMS:
                synonyms = self.MEDICAL_SYNONYMS[clean_word]
                new_word = random.choice(synonyms)
                result.append(word.replace(clean_word, new_word))
            else:
                result.append(word)
        
        return " ".join(result)
    
    def forward(
        self,
        embedding: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        """
        Args:
            embedding: (B, seq_len, embed_dim) or (B, embed_dim) - CLIP embedding
            text: Original text string (for synonym substitution)
        
        Returns:
            Tuple of (perturbed_embedding, perturbed_text)
        """
        if not self.training:
            return embedding, text
        
        perturbed_embedding = embedding
        perturbed_text = text
        
        if embedding is not None:
            perturbed_embedding = self.add_embedding_noise(embedding)
        
        if text is not None:
            perturbed_text = self.substitute_synonyms(text)
        
        return perturbed_embedding, perturbed_text


class PromptPerturbation(nn.Module):
    """
    Combined perturbation module for all prompt types.
    Wraps BBoxPerturbation and PointPerturbation.
    """
    
    def __init__(
        self,
        # BBox params
        sigma_b: float = 5.0,
        gamma_range: Tuple[float, float] = (-2.0, 2.0),
        rotation_range: float = 3.0,
        apply_rotation: bool = True,
        # Point params
        sigma_p: float = 3.0,
        q_flip: float = 0.05,
        # Mask params
        dilate_radius: int = 1,
        erode_radius: int = 1,
        warp_strength: float = 0.02,
        p_dilate: float = 0.2,
        p_erode: float = 0.2,
        p_warp: float = 0.2,
        min_area_ratio: float = 0.3,
        # Text params
        sigma_t: float = 0.1,
        p_noise: float = 0.5,
        p_synonym: float = 0.3,
        embed_dim: int = 512,
    ):
        super().__init__()
        self.bbox_perturbation = BBoxPerturbation(
            sigma_b=sigma_b,
            gamma_range=gamma_range,
            rotation_range=rotation_range,
            apply_rotation=apply_rotation,
        )
        self.point_perturbation = PointPerturbation(
            sigma_p=sigma_p,
            q_flip=q_flip,
        )
        self.text_perturbation = TextPerturbation(
            sigma_t=sigma_t,
            p_noise=p_noise,
            p_synonym=p_synonym,
            embed_dim=embed_dim,
        )
        self.mask_perturbation = MaskPerturbation(
            dilate_radius=dilate_radius,
            erode_radius=erode_radius,
            warp_strength=warp_strength,
            p_dilate=p_dilate,
            p_erode=p_erode,
            p_warp=p_warp,
            min_area_ratio=min_area_ratio,
        )

    @classmethod
    def from_config(
        cls,
        mppg_config: MPPGConfig,
        **overrides,
    ) -> "PromptPerturbation":
        """Build the perturbation wrapper from MPPGConfig with optional overrides."""
        params = {
            "sigma_b": mppg_config.sigma_b,
            "gamma_range": mppg_config.gamma_range,
            "rotation_range": mppg_config.rotation_range,
            "sigma_p": mppg_config.sigma_p,
            "q_flip": mppg_config.q_flip,
            "dilate_radius": mppg_config.dilate_radius,
            "erode_radius": mppg_config.erode_radius,
            "warp_strength": mppg_config.warp_strength,
            "sigma_t": mppg_config.sigma_t,
        }
        params.update(overrides)
        return cls(**params)

    def forward(
        self,
        bbox: Optional[torch.Tensor] = None,
        points: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        mask: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            bbox: (B, N, 4) bounding boxes
            points: (B, N, 2) point coordinates
            point_labels: (B, N) point labels
        
        Returns:
            Dict with perturbed prompts
        """
        result = {}
        
        if bbox is not None:
            result["bbox"] = self.bbox_perturbation(bbox)
        
        if points is not None:
            perturbed_points, perturbed_labels = self.point_perturbation(
                points, point_labels
            )
            result["points"] = perturbed_points
            result["point_labels"] = perturbed_labels
        if text is not None or text_embeddings is not None:
            perturbed_embedding, perturbed_text = self.text_perturbation(
                embedding=text_embeddings, text=text
            )
            result["text_embedding"] = perturbed_embedding
            result["text"] = perturbed_text
            
        if mask is not None:
            result["mask"] = self.mask_perturbation(mask)
            
        return result

# ============== Test ==============
# if __name__ == "__main__":
#     # Test BBox Perturbation
#     print("=" * 50)
#     print("Testing BBoxPerturbation")
#     print("=" * 50)
    
#     bbox_perturb = BBoxPerturbation(sigma_b=5.0, rotation_range=3.0)
#     bbox_perturb.train()
    
#     bbox = torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32)
#     print(f"Original bbox: {bbox}")
    
#     perturbed_bbox = bbox_perturb(bbox)
#     print(f"Perturbed bbox: {perturbed_bbox}")
    
#     # Test Point Perturbation
#     print("\n" + "=" * 50)
#     print("Testing PointPerturbation")
#     print("=" * 50)
    
#     point_perturb = PointPerturbation(sigma_p=3.0, q_flip=0.1)
#     point_perturb.train()
    
#     points = torch.tensor([[[150, 150], [120, 180]]], dtype=torch.float32)
#     labels = torch.tensor([[1, 0]], dtype=torch.long)
    
#     print(f"Original points: {points}")
#     print(f"Original labels: {labels}")
    
#     perturbed_points, perturbed_labels = point_perturb(points, labels)
#     print(f"Perturbed points: {perturbed_points}")
#     print(f"Perturbed labels: {perturbed_labels}")
    
#     # Test Combined
#     print("\n" + "=" * 50)
#     print("Testing PromptPerturbation (Combined)")
#     print("=" * 50)
    
#     prompt_perturb = PromptPerturbation()
#     prompt_perturb.train()
    
#     result = prompt_perturb(
#         bbox=torch.tensor([[[100, 100, 200, 200]]], dtype=torch.float32),
#         points=torch.tensor([[[150, 150]]], dtype=torch.float32),
#         point_labels=torch.tensor([[1]], dtype=torch.long),
#     )
