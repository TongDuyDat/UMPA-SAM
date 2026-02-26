import torch
from .consistency_loss import MultiPromptConsistencyLoss
from typing import Dict

class ComposerLoss(torch.nn.Module):
    def __init__(self, config_loss: Dict[str, float], *args, **kwargs):
        super(ComposerLoss, self).__init__()
        self.dice_loss = MultiPromptConsistencyLoss(config_loss['consistency_loss_weight'])
        self.consistency_loss = MultiPromptConsistencyLoss(config_loss['consistency_loss_weight'])
        self.config_loss = config_loss
        
    def forward(self, pred_masks, gt_masks, perturbed_masks):
        """The full training loss becomes:
            Ltotal = Lseg + λconLcon + λregLreg
        """
        seg_loss = self.dice_loss(pred_masks, gt_masks)
        consistency_loss = self.consistency_loss(perturbed_masks)
        total_loss = seg_loss + self.config_loss['consistency_loss_weight'] * consistency_loss
        
        return {
            "total_loss": total_loss,
            "seg_loss": seg_loss,
            "consistency_loss": consistency_loss,
        }
    
    def __repr__(self):
        return f"ComposerLoss(dice_loss={self.dice_loss}, consistency_loss={self.consistency_loss}, config_loss={self.config_loss})"