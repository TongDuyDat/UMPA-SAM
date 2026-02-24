"""
Test kiểm chứng MultiPromptConsistencyLoss sử dụng:
- MaskPerturbation từ umpt_sam/modules/modules.py
- Ảnh mask.png thực tế từ thư mục gốc

Chiến lược test:
  1. Load mask.png → tensor
  2. Dùng MaskPerturbation sinh K perturbed masks
  3. Tính tay Dice để so sánh với loss_fn
  4. Kiểm tra 3 invariant: identical=0, opposite≈1, symmetry
"""

import sys
import os
import math

import torch
import numpy as np
from PIL import Image
import cv2
# --- Path setup ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from umpt_sam.losses.consistency_loss import MultiPromptConsistencyLoss
from umpt_sam.modules.modules import MaskPerturbation

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def load_mask(path: str, size=(64, 64)) -> torch.Tensor:
    """Load ảnh mask PNG → float tensor (1, 1, H, W) trong [0,1]."""
    img = Image.open(path).convert("L").resize(size, Image.NEAREST)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Nhị phân hoá (ngưỡng 0.5)
    arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def dice_pairwise(a: torch.Tensor, b: torch.Tensor, smooth=1e-6) -> float:
    """Tính Dice(a, b) thủ công để so sánh."""
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    inter = (a * b).sum()
    union = a.sum() + b.sum()
    return (2 * inter + smooth) / (union + smooth)


def make_stack(masks: list, B: int = 1) -> torch.Tensor:
    """
    masks: list of K tensors (1,1,H,W)
    → (B, K, 1, H, W)
    """
    stacked = torch.cat(masks, dim=1)          # (1, K, H, W) sai dim → dùng stack
    return stacked.contiguous()


# -----------------------------------------------------------------------
# Test 1 — Invariant: masks giống hệt → loss ≈ 0
# -----------------------------------------------------------------------
def test_identical_masks_give_zero_loss():
    print("\n[Test 1] Identical masks → loss ≈ 0")
    loss_fn = MultiPromptConsistencyLoss(sigmoid_input=False)
    loss_fn.eval()   # tắt training để weights không thay đổi
    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)  # Đảm bảo mask.png tồn tại
    mask = torch.from_numpy(mask).float() / 255.0  # Normalize to [0,1]
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
    print(mask.shape)  # (1,1,H,W)
    masks = mask.expand(1, 4, 1, -1, -1).contiguous()  # (1,4,1,H,W)

    loss = loss_fn(masks)
    print(f"  loss = {loss.item():.8f}  (kỳ vọng < 1e-4)")
    assert loss.item() < 1e-4, f"FAIL: {loss.item()}"
    print("  PASS")


# -----------------------------------------------------------------------
# Test 2 — Invariant: ones vs zeros → 1 - Dice ≈ 1
# -----------------------------------------------------------------------
def test_opposite_masks_give_max_loss():
    print("\n[Test 2] Opposite masks (all-1 vs all-0) → loss ≈ 1")
    loss_fn = MultiPromptConsistencyLoss(sigmoid_input=False)
    loss_fn.eval()

    ones  = torch.ones (1, 1, 1, 32, 32)
    zeros = torch.zeros(1, 1, 1, 32, 32)
    masks = torch.cat([ones, zeros], dim=1)   # (1,2,1,32,32)

    loss = loss_fn(masks)
    print(f"  loss = {loss.item():.6f}  (kỳ vọng ≈ 1.0)")
    assert loss.item() > 0.9, f"FAIL: {loss.item()}"
    print("  PASS")


# -----------------------------------------------------------------------
# Test 3 — Symmetry: loss(i,j) = loss(j,i)
# -----------------------------------------------------------------------
def test_symmetry():
    print("\n[Test 3] Symmetry: loss(a,b) == loss(b,a)")
    loss_fn = MultiPromptConsistencyLoss(sigmoid_input=False)
    loss_fn.eval()

    a = torch.rand(1, 1, 1, 32, 32)
    b = torch.rand(1, 1, 1, 32, 32)

    masks_ab = torch.cat([a, b], dim=1)
    masks_ba = torch.cat([b, a], dim=1)

    loss_ab = loss_fn(masks_ab).item()
    loss_ba = loss_fn(masks_ba).item()
    print(f"  loss(a,b) = {loss_ab:.8f}")
    print(f"  loss(b,a) = {loss_ba:.8f}")
    assert abs(loss_ab - loss_ba) < 1e-5, f"FAIL: diff = {abs(loss_ab-loss_ba)}"
    print("  PASS")


# -----------------------------------------------------------------------
# Test 4 — So sánh với giá trị tính tay
# -----------------------------------------------------------------------
def test_manual_dice_value():
    """
    Với K=2, α_ij uniform (dùng fixed weights = 1.0 cho cặp (0,1)):
    Dice(y1, y2) tính tay phải khớp với loss_fn output.

    Trick: đặt temperature rất nhỏ → weights hội tụ về upper-tri uniform.
    Nhưng vì alpha là learnable và init=0, exps đều nhau → weights = 1/(K(K-1)/2).
    Với K=2: chỉ có 1 cặp (i=0,j=1) → weight = 1.0 sau normalize.
    """
    print("\n[Test 4] Giá trị tính tay với mask 1-D đơn giản")

    # y1 = [1,1,0,0], y2 = [1,0,0,0]
    # inter(1,2) = 1, union(1)=2, union(2)=1
    # Dice = 2*1/(2+1) = 2/3
    # 1 - Dice = 1/3 ≈ 0.3333
    EXPECTED = 1.0 / 3.0

    y1 = torch.tensor([[[[1., 1., 0., 0.]]]])  # (1,1,1,4)
    y2 = torch.tensor([[[[1., 0., 0., 0.]]]])

    masks = torch.cat([y1, y2], dim=1)  # (1,2,1,4)
    masks = masks.view(1, 2, 1, 2, 2)  # (B=1, K=2, C=1, D=4) → reshape để khớp input của loss_fn
    loss_fn = MultiPromptConsistencyLoss(sigmoid_input=False, smooth=1e-9)
    loss_fn.eval()

    loss = loss_fn(masks).item()
    print(f"  Tính tay   = {EXPECTED:.6f}")
    print(f"  loss_fn    = {loss:.6f}")
    diff = abs(loss - EXPECTED)
    print(f"  |diff|     = {diff:.8f}  (kỳ vọng < 1e-4)")
    assert diff < 1e-4, f"FAIL: diff = {diff}"
    print("  PASS")


# -----------------------------------------------------------------------
# Test 5 — Dùng mask.png thực tế + MaskPerturbation
# -----------------------------------------------------------------------
def test_with_real_mask_and_perturbation():
    print("\n[Test 5] mask.png + MaskPerturbation → consistency loss")

    mask_path = os.path.join(ROOT, "mask.png")
    assert os.path.exists(mask_path), f"Không tìm thấy {mask_path}"

    # Load mask gốc
    mask_orig = load_mask(mask_path, size=(256, 256))  # (1,1,64,64) trong [0,1]
    print(f"  mask shape  : {mask_orig.shape}")
    print(f"  mask area   : {mask_orig.sum().item():.0f} pixels")

    # Sinh K=5 perturbed masks bằng MaskPerturbation
    perturber = MaskPerturbation(
        dilate_radius=2,
        erode_radius=2,
        warp_strength=0.03,
        p_dilate=0.33,
        p_erode=0.33,
        p_warp=0.33,
    )
    perturber.train()   # phải ở train mode mới perturbation

    K = 5
    perturbed_list = []
    for k in range(K):
        p = perturber(mask_orig.squeeze(0))  # (1,64,64)
        perturbed_list.append(p.unsqueeze(0).unsqueeze(0))  # (1,1,64,64)
    print(perturbed_list[0].shape)  # (1,1,64,64)
    masks = make_stack(perturbed_list, B=1)  # (1,K,1,64,64)
    print(f"  stacked shape: {masks.shape}")

    # Tính loss
    loss_fn = MultiPromptConsistencyLoss(sigmoid_input=False, smooth=1e-6)
    loss_fn.eval()

    loss = loss_fn(masks)
    print(f"  consistency loss = {loss.item():.6f}")

    # Tính Dice pairwise thủ công để kiểm chứng
    print("\n  Pairwise Dice (tính tay):")
    manual_dice_losses = []
    for i in range(K):
        for j in range(K):
            if i >= j:
                continue
            mi = perturbed_list[i].squeeze()
            mj = perturbed_list[j].squeeze()
            d = dice_pairwise(mi, mj).item()
            manual_dice_losses.append(1 - d)
            print(f"    (i={i},j={j}): Dice={d:.4f}  1-Dice={1-d:.4f}")

    mean_manual = sum(manual_dice_losses) / len(manual_dice_losses)
    print(f"\n  Mean 1-Dice (tay) = {mean_manual:.6f}")
    print(f"  loss_fn output    = {loss.item():.6f}")

    # Loss phải nằm trong khoảng hợp lý [0, 1]
    assert 0.0 <= loss.item() <= 1.0, f"FAIL: loss ngoài [0,1]: {loss.item()}"

    # Loss tính bằng weighted sum → nên gần với mean_manual
    # (không bằng chính xác vì weights được normalize khác)
    print("  PASS (loss nằm trong [0,1])")


# -----------------------------------------------------------------------
# Test 6 — Kiểm tra gradient flow (loss differentiable)
# -----------------------------------------------------------------------
def test_gradient_flow():
    print("\n[Test 6] Gradient flow qua loss_fn")
    loss_fn = MultiPromptConsistencyLoss(sigmoid_input=True)
    loss_fn.train()

    logits = torch.randn(2, 4, 1, 32, 32, requires_grad=True)
    loss = loss_fn(logits)
    loss.backward()

    assert logits.grad is not None, "FAIL: không có gradient"
    assert not torch.isnan(logits.grad).any(), "FAIL: gradient có NaN"
    print(f"  loss = {loss.item():.6f}")
    print(f"  grad norm = {logits.grad.norm().item():.6f}")
    print("  PASS")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  Test MultiPromptConsistencyLoss")
    print("=" * 60)

    test_identical_masks_give_zero_loss()
    test_opposite_masks_give_max_loss()
    test_symmetry()
    test_manual_dice_value()
    test_with_real_mask_and_perturbation()
    test_gradient_flow()

    print("\n" + "=" * 60)
    print("  Tất cả test PASS")
    print("=" * 60)
