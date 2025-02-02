import torch.nn as nn

from gelu import GELU

class FeedForward(nn.Module):
    # we are going to use a configuration dict here to avoid have to pass in random looking 
    # parameters
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


