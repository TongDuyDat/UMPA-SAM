import torch
from torch import nn
from typing import Dict

from ..config.model_config import UPFEConfig

# Unified Prompt Fusion Encoder
class UnifiedPromptFusionEncoder(nn.Module):
    @classmethod
    def from_config(
        cls,
        upfe_config: UPFEConfig,
        scoting_network: nn.Module = None,
    ) -> "UnifiedPromptFusionEncoder":
        """Build the encoder from a typed UPFEConfig dataclass."""
        return cls(
            scoting_network=scoting_network,
            embed_dim=upfe_config.embed_dim,
            scoting_network_hidden_dim=upfe_config.scoring_hidden_dim,
        )

    def __init__(self,
        scoting_network: nn.Module = None,
        embed_dim = 256,
        scoting_network_hidden_dim = 256,
    ):
        super(UnifiedPromptFusionEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.scoting_network_hidden_dim = scoting_network_hidden_dim

        if scoting_network is None:
            self.scoting_network = self.build_scoting_network()
        else:
            self.scoting_network = scoting_network

    def build_scoting_network(self):
        return nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim//4),
            nn.ReLU(),
            nn.Linear(self.embed_dim//4, self.scoting_network_hidden_dim),
        )

    def forward(self, embeddings: Dict[str, torch.Tensor], return_weights=False):
        """
            Args:
                embeddings: Dict[str, torch.Tensor] = {
                    "box_embeddings": torch.Tensor,
                    "point_embeddings": torch.Tensor,
                    "mask_embeddings": torch.Tensor,
                    "text_embeddings": torch.Tensor,
                }
        """
        scores = []
        embeds = []
        for embed_name, embed in embeddings.items():
            if embed is None:
                continue
            if embed.shape[-1] != self.embed_dim:
                embed = embed.permute(0, 2, 1)
            embeds.append(embed)
            adapt_weight = self.scoting_network(embed)
            scores.append(adapt_weight)
        scores = torch.cat(scores, dim=1)
        weights = torch.softmax(scores, dim=1) 
        embeds = torch.cat(embeds, dim=1)
        e_fused = torch.sum(weights * embeds, dim=1)

        if return_weights:
            return e_fused, weights
        
        return e_fused

# if __name__ == "__main__":
    
#     box_embeddings = torch.randn(1, 20, 256)
#     point_embeddings = torch.randn(1, 10, 256)
#     mask_embeddings = torch.randn(1, 256, 64*64)
#     text_embeddings = torch.randn(1, 100, 256)
#     embeddings = {
#         "box_embeddings": box_embeddings,
#         "point_embeddings": point_embeddings,
#         "mask_embeddings": mask_embeddings,
#         "text_embeddings": text_embeddings,
#     }
#     upf = UnifiedPromptFusionEncoder()
#     e_fused, weights = upf(embeddings, return_weights=True)
#     print(e_fused.shape)
#     print(weights.shape)
