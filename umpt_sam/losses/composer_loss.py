import torch
from .consistency_loss import MultiPromptConsistencyLoss
from .dice_loss import DiceLoss
from .regularization_loss import RegularizationLoss
from typing import Dict

class ComposerLoss(torch.nn.Module):
    def __init__(self, config_loss: Dict[str, float], *args, **kwargs):
        super(ComposerLoss, self).__init__()
        self.dice_loss = DiceLoss(config_loss['dice_loss_weight'])
        self.consistency_loss = MultiPromptConsistencyLoss(config_loss['consistency_loss_weight'])
        self.config_loss = config_loss
        self.regularization_loss = RegularizationLoss(config_loss['regularization_loss_weight'])
    def forward(self, pred_masks, gt_masks, perturbed_masks, prompt_weights=None):
        """The full training loss becomes:
            Ltotal = Lseg + λconLcon + λregLreg
        """
        seg_loss = self.dice_loss(pred_masks, gt_masks)
        # consistency_loss = self.consistency_loss(perturbed_masks)
        if perturbed_masks is not None:
            consistency_loss = self.consistency_loss(perturbed_masks)
        else:
            # Nếu ở Phase 1 hoặc 2 không truyền perturbed_masks lambda_con =0, gán loss = 0
            consistency_loss = torch.tensor(0.0, device=pred_masks.device)

        if prompt_weights is not None:
            regularization_loss = self.regularization_loss(prompt_weights)
            total_loss = seg_loss + self.config_loss['consistency_loss_weight'] * consistency_loss + self.config_loss['regularization_loss_weight'] * regularization_loss
        else:
            total_loss = seg_loss + self.config_loss['consistency_loss_weight'] * consistency_loss
        total_loss = seg_loss + self.config_loss['consistency_loss_weight'] * consistency_loss
        
        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "consistency_loss": consistency_loss,
            "regularization_loss": regularization_loss if prompt_weights is not None else None,
        }
    
    def __repr__(self):
        return f"ComposerLoss(dice_loss={self.dice_loss}, consistency_loss={self.consistency_loss}, regularization_loss={self.regularization_loss}, config_loss={self.config_loss})"