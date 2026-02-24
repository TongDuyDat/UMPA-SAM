
import torch

class DiceLoss(torch.nn.Module):
    """Dice Loss for binary segmentation.

    Based on Dice coefficient from: V-Net: Fully Convolutional Neural Networks
    for Volumetric Medical Image Segmentation (Milletari et al., 2016)
    """

    def __init__(self, smooth=1e-6, squared_union=True,):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.squared_union = squared_union

    def forward(self, predicts, targets):
        """_summary_

        Args:
            predicts (torch.Tensor): Predicted masks (B, 1, H, W) in [0,1]
            targets (torch.Tensor): Ground truth masks (B, 1, H, W) in [0,1]

        Returns:
            torch.Tensor: Dice loss value
        """
        predicts = predicts.view(predicts.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = torch.einsum("bc, bc -> b", predicts, targets)
        if self.squared_union:
            union = torch.einsum("bt -> b", predicts**2) + torch.einsum("bt -> b", targets**2)
        else: 
            union = predicts.sum(dim=1) + targets.sum(dim=1)
        dsc = (2 * intersection + self.smooth) / (union + self.smooth)
        dice = 1 - dsc  # 1 - dice score
        return dice.mean()
