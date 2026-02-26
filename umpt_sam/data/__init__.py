from .polyp_dataset import PolypDataset, collate_fn
from .transforms import TransformConfig, build_transforms

__all__ = ["PolypDataset", "collate_fn", "TransformConfig", "build_transforms"]

"""
Example usage:
from umpt_sam.data import PolypDataset, DatasetConfig, TransformConfig, collate_fn
from torch.utils.data import DataLoader

cfg = DatasetConfig.kvasir_seg(
    root="data/data_benmarks/kvasir-seg",
    transform_cfg=TransformConfig(image_size=1024),
)
dataset = PolypDataset(cfg, phase="train")
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
# batch["image"]  → (4, 3, 1024, 1024)
# batch["bbox"]   → (4, 4)              → PromptPerturbation.bbox_perturbation
# batch["points"] → (4, N_max, 2)       → PromptPerturbation.point_perturbation
# batch["coarse_mask"] → (4, 1, 1024, 1024) → PromptPerturbation.mask_perturbation
# batch["text"]   → List[str]           → TextPerturbation
# """