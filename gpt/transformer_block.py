import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward import FeedForward
from gelu import GELU

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadAttention(d_in = cfg["emb_dim"],
                                       d_out = cfg["emb_dim"],
                                       context_length = cfg["context_length"],
                                       dropout = cfg["drop_rate"],
                                       num_heads = cfg["n_heads"],
                                       qkv_bias = cfg["qkv_bias"])
        self.dropout = nn.Dropout(cfg["drop_rate"])
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.feed_forward = FeedForward(cfg)

    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + shortcut

        return x
        
#####
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

torch.manual_seed(123)

block = TransformerBlock(GPT_CONFIG_124M)

x = torch.randn(2, 4, 768) #batch size 2, each with 4 input tokens, 768 embedded dimensions

output = block(x)

print(x.shape)
print(output.shape)
