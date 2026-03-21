"""Diagnostic: check UPFE gradient flow with real SAM3 model."""
import sys
sys.path.insert(0, ".")
import torch
from umpt_sam.umpa_model import UMPAModel
from umpt_sam.config.model_config import UMPAModelConfig, UPFEConfig, MPPGConfig

cfg = UMPAModelConfig(
    sam_checkpoint="model_trained/sam3.pt",
    embed_dim=256, text_embed_dim=512,
    freeze_image_encoder=True,
    upfe=UPFEConfig(scoring_hidden_dim=256),
    mppg=MPPGConfig(),
)
model = UMPAModel.from_config(model_config=cfg).to("cuda")
model.train()

print("=== requires_grad status ===")
pe_grad = any(p.requires_grad for p in model.prompt_encoder.parameters())
print(f"prompt_encoder: {pe_grad}")
upfe_grad = any(p.requires_grad for p in model.upfe_encoder.parameters())
print(f"upfe_encoder: {upfe_grad}")
ie_grad = any(p.requires_grad for p in model.image_encoder.parameters())
print(f"image_encoder: {ie_grad}")
md_grad = any(p.requires_grad for p in model.sam_mask_decoder.parameters())
print(f"mask_decoder: {md_grad}")

print("\n=== Forward + backward ===")
image = torch.randn(1, 3, 1008, 1008, device="cuda")
boxes = torch.tensor([[[100, 100, 400, 400]]], dtype=torch.float32, device="cuda")
out = model(image=image, boxes=boxes)
loss = out["pred_masks"].sum()
loss.backward()

print("\n=== UPFE gradients ===")
for name, p in model.upfe_encoder.named_parameters():
    g = p.grad
    gsum = g.abs().sum().item() if g is not None else 0
    print(f"  {name}: req_grad={p.requires_grad}, grad_sum={gsum:.6f}")

print("\n=== Mask decoder gradients (first 5) ===")
for i, (name, p) in enumerate(model.sam_mask_decoder.named_parameters()):
    if i >= 5:
        break
    g = p.grad
    gsum = g.abs().sum().item() if g is not None else 0
    print(f"  {name}: req_grad={p.requires_grad}, grad_sum={gsum:.6f}")

print("\n=== text_projection gradients ===")
for name, p in model.text_projection.named_parameters():
    g = p.grad
    gsum = g.abs().sum().item() if g is not None else 0
    print(f"  {name}: req_grad={p.requires_grad}, grad_sum={gsum:.6f}")
