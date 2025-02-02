## try out the helpers code


import tiktoken

from gpt import GPTModule
from helpers import generate_text_simple
from helpers import text_to_token_ids
from helpers import token_ids_to_text


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 256,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

model = GPTModule(GPT_CONFIG_124M)

start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(model,
                                 idx = text_to_token_ids(start_context, tokenizer),
                                 max_new_tokens = 10,
                                 context_size = 256)

text = token_ids_to_text(token_ids, tokenizer)

print(text)
    
