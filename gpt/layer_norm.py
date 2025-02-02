import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    #embedded_dim represents the number of embedded dimensions
    def __init__(self, embedded_dim):
        super().__init__()
        self.epsilon = 1e-5 #see below. avoids a division by 0.
        self.scale = nn.Parameter(torch.ones(embedded_dim))
        self.shift = nn.Parameter(torch.zeros(embedded_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm = (x - mean)/torch.sqrt(var + self.epsilon)
        scaled_shifted_norm = (norm * self.scale) + self.shift
        return scaled_shifted_norm


batch = torch.randn(2, 5)
l = LayerNorm(batch.shape[-1])
out = l(batch)
print(out)

torch.set_printoptions(sci_mode=False)
print(out.mean(dim=-1, keepdim=True), out.var(dim=-1, keepdim=True, unbiased=False))
