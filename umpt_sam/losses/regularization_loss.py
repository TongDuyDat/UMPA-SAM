import torch

class RegularizationLoss(torch.nn.Module):
    """Lreg is a minor regularization term applied to prompt perturbations to prevent 
        collapse.
        L_reg = Σ_{t}w_tlog(w_t + ε)
    where w_t are the normalized weights of the perturbed prompts.
    """

    def __init__(self, epsilon: float = 1e-8):
        super(RegularizationLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute regularization loss on prompt weights.

        Args:
            weights (torch.Tensor): Normalized weights of perturbed prompts (B, K, D)

        Returns:
            torch.Tensor: Regularization loss value
        """
        # L_reg = Σ_{t} w_t log(w_t + ε)
        reg_loss = -torch.sum(weights * torch.log(weights + self.epsilon), dim=1)
        return reg_loss.mean()