import torch
from torch import nn

class MultiPromptConsistencyLoss(nn.Module):
    """Multi-Prompt Consistency Loss for robust segmentation.

    Enforces prediction stability across K perturbed prompts by minimizing
    pairwise Dice disagreement:

        L_con = Σ_{i≠j} α_ij (1 - Dice(Ŷ^(i), Ŷ^(j)))

    where Ŷ^(k) = Φ(I, P̃^(k)) are masks from perturbed prompts.

    Mathematical Properties:
    - Symmetric: L(Ŷ^(i), Ŷ^(j)) = L(Ŷ^(j), Ŷ^(i))
    - Non-negative: L_con ≥ 0
    - Zero iff all masks are identical

    Complexity: O(K² · B · H · W) for K prompts, batch B, spatial dims H×W
    """

    def __init__(
        self,
        temperature: float = 0.1,
        smooth: float = 1e-6,
        sigmoid_input: bool = True,
    ):
        """
        Args:
            weighting: Pairwise weight scheme
                - "uniform": α_ij = 1 / (K(K-1))
                - "softmax": α_ij ∝ exp(-||Ŷ^(i) - Ŷ^(j)||₂ / τ)
                - "learnable": Trainable α matrix (K×K upper triangular)
            temperature: τ for softmax weighting (lower → sharper)
            smooth: ε for numerical stability in Dice
            sigmoid_input: If True, apply sigmoid to inputs (for logits)
        """
        super().__init__()
        self.temperature = temperature
        self.smooth = smooth
        self.sigmoid_input = sigmoid_input
        # For learnable weights (initialized lazily based on K)
        self._alpha = None

    def _compute_weights(self, K: int, device) -> torch.Tensor:
        # 1. Khởi tạo tham số học được
        if self._alpha is None or self._alpha.size(0) != K:
            self._alpha = nn.Parameter(torch.zeros(K, K, device=device))
            self.register_parameter("alpha_param", self._alpha)

        # 2. Tạo mặt nạ tam giác trên để LOẠI BỎ i=j (diagonal=1)
        mask = torch.triu(torch.ones(K, K, device=device), diagonal=1)
        
        # 3. Tính trọng số: Chỉ tính trên các vùng i < j
        # Dùng Global Softmax để tổng TẤT CẢ alpha_ij (với i < j) bằng 1
        exps = torch.exp(self._alpha / self.temperature) * mask
        weights = exps / (exps.sum() + 1e-8)
        
        return weights

    def forward(self, predicted_masks: torch.Tensor) -> torch.Tensor:
        """Compute multi-prompt consistency loss.

        Args:
            predicted_masks: (B, K, 1, H, W) or list of K tensors (B, 1, H, W)
                Segmentation masks from K different perturbed prompts
                Can be logits (if sigmoid_input=True) or probabilities

        Returns:
            loss: Scalar consistency loss averaged over batch

        Example:
            >>> loss_fn = MultiPromptConsistencyLoss()
            >>> masks = torch.rand(4, 5, 1, 256, 256)  # B=4, K=5 prompts
            >>> loss = loss_fn(masks)
        """
        # Handle list input
        if isinstance(predicted_masks, (list, tuple)):
            predicted_masks = torch.stack(predicted_masks, dim=1)

        B, K, C, H, W = predicted_masks.shape
        assert C == 1, f"Expected single-channel masks, got C={C}"

        if self.sigmoid_input:
            predicted_masks = torch.sigmoid(predicted_masks)
            
        predicted_masks = predicted_masks.view(B, K, C*H*W)
        inter = torch.einsum("bid, bjd-> bij", predicted_masks, predicted_masks)
        
        union = torch.einsum("bkd-> bk", predicted_masks)
        union = union.unsqueeze(1) + union.unsqueeze(2)  # (B, K, K)
        dice_score = (2 * inter + self.smooth) / (union + self.smooth) 

        dice_loss = 1 - dice_score 
        # --- Áp dụng trọng số loại bỏ i=j ---
        weights = self._compute_weights(K, predicted_masks.device)
        # Nhân dice_loss (có i=j bằng 0) với weights (cũng có i=j bằng 0)
        # Kết quả là tổng của các cặp i < j duy nhất
        total_loss = torch.einsum("bij, ij -> b", dice_loss, weights) 

        return total_loss.mean()

# if __name__ == "__main__":
#     loss_fn = MultiPromptConsistencyLoss(sigmoid_input=True)

    
#     # 1. Thử với Logits ngẫu nhiên (randn thay vì rand)
#     # randn có cả giá trị âm và dương, sau sigmoid sẽ ra đủ dải [0, 1]
#     dice_loss = DiceLoss()
#     masks_random = torch.randn(4, 5, 1, 256, 256) 
#     masks_random_for_dice = torch.randn(4, 1, 256, 256)
#     targets_random = torch.randn(4, 1,256, 256)
#     print(dice_loss(masks_random_for_dice, targets_random))
#     loss_rand = loss_fn(masks_random)
#     print(f"Loss với dữ liệu ngẫu nhiên (nên cao): {loss_rand.item():.4f}")

#     # 2. Thử với các mặt nạ giống hệt nhau (Consistency hoàn hảo)
#     mask_single = torch.randn(4, 1, 1, 256, 256)
#     masks_identical = mask_single.expand(-1, 5, -1, -1, -1)
#     loss_id = loss_fn(masks_identical)
#     print(f"Loss với dữ liệu giống hệt (phải gần 0): {loss_id.item():.4f}")