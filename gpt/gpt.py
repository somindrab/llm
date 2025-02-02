import torch
import torch.nn as nn

from transformer_block import TransformerBlock
from layer_norm import LayerNorm
from helpers import generate_text_simple

class GPTModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.positional_embedding = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.dropout = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
                                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])

        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape

        tok_emb = self.token_embedding(in_idx)
        pos_emb = self.positional_embedding(
            torch.arange(seq_len, device=in_idx.device)
        )

        x = tok_emb + pos_emb

        x = self.dropout(x)

        x = self.trf_blocks(x)

        x = self.final_norm(x)

        logits = self.out_head(x)

        return logits

"""
### Let's try this out

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

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2))) #batch is a list
batch = torch.stack(batch, dim=0) #converts from a list of two tensors to a tensor of shape 2,4

gpt = GPTModule(GPT_CONFIG_124M)

out = gpt(batch)

print(out.shape) #[2, 4, 50257] i.e. 2 batches, 4 tokens each, 50257 dimensions per token

context = "Hello, I am"

encoded_context = tokenizer.encode(context)

#print(encoded_context) #just a list

encoded_context_tensor = torch.tensor(encoded_context)
#print(encoded_context_tensor) #just a tensor with the same info as the list

#we want this to look like (batch_size, tokens)
encoded_context_tensor = encoded_context_tensor.unsqueeze(0)
#print(encoded_context_tensor)
#print(encoded_context_tensor.shape)

gpt.eval()
out = generate_text_simple(gpt, encoded_context_tensor, 6, context_size=GPT_CONFIG_124M["context_length"])

print(out)

decoded_text = tokenizer.decode(out.squeeze(0).tolist())

print(decoded_text)

#'Hello, I am Featureiman Byeswickattribute argue'
# classic

"""
