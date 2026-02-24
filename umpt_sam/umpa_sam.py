import torch
from torch import nn
from typing import Dict

class UMPASAM(nn.Module):
    def __init__(self, ):
        super(UMPASAM, self).__init__()
        self.perturbation = None
        self.image_sam_encoder = None
        self.prompt_fusion_encoder = None
        self.cross_attn_integration = None
        self.mask_decoder = None
    
    def consistency_loss(self, pred_mask, gt_mask):
        """
        Y (k) = Φ(I, P(k)), k = 1, . . . ,K,
        L_con = sum (αij(1 - Dice(Y (i), Y (j)))) with i̸=j
        Args:
            pred_mask: (B, 1, H, W)
            gt_mask: (B, 1, H, W
        Returns:
            consistency_loss: (B,)
        """
        
