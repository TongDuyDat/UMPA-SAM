

from typing import Dict
from torch import nn

class UMPAModel(nn.Module):
    def __init__(self, embed_dim, image_size):
        super(UMPAModel, self).__init__()
        self.sam_encoder = None
        self.sam_mask_decoder = None
        self.upfe_encoder = None
        self.cross_attn_integration = None
    def forward(self, image, embeddings: Dict[str, torch.Tensor]):
        image_encode = self.sam_encoder(image)
        upfe_fusion_embs = self.upfe_encoder(embeddings)
        cross_attn_integration = self.cross_attn_integration(image_encode, upfe_fusion_embs)
        mask = self.sam_mask_decoder(cross_attn_integration)
        return mask
