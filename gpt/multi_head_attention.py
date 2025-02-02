import torch
import torch.nn as nn

## So here, the d_out has to be a multiple of the number of heads.
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout,
                 num_heads, qkv_bias=False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        #The number of heads per dimension
        self.head_dim = self.d_out // self.num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        #print(f"W_key weight shape: {self.W_key.weight.shape}")
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #print(f"keys.shape: {keys.shape}")
        
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        
        #print(f"keys.shape after splitting into heads: {keys.shape}")

        #Now here, we are going to "rearrange" the matrices such that we go from
        #[b, num_tokens, num_heads, head_dim] -> [b, num_heads, num_tokens, head_dim]
        #and this should make a lot of sense because after all, we are trying to process
        #multiple heads here, and so arranging this in that hierarchy is what we need to do

        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)

        #and now really business as usual. We want to compute the attn weights and context
        #vectors, just that we need to remain hyper aware that our matrices are hierarchically
        #arranged as batches -> heads -> tokens -> embedded vectors
        attn_scores = queries @ keys.transpose(2,3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        #scaled dot product
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        #apply the dropout if there is one specified
        attn_weights = self.dropout(attn_weights)

        #calculate the context vectors
        context_vectors = attn_weights @ values

        #Now, we want to go back to how we desire the context vectors, i.e.,
        #(b, num_tokens, vectors) where all of the vectors from the multiple heads are
        #"combined" to see the uniform result of multihead processing.
        #To get there, we need to start rearranging the hierarchy 

        context_vectors = context_vectors.transpose(1,2)
        #so now this will become (b, tokens, heads, vectors)

        context_vectors = context_vectors.contiguous().view(b, num_tokens, self.d_out)
        #so now this will become (b, tokens, vectors)

        #apply a linear projection that will be useful when this class is used in training
        context_vectors = self.out_proj(context_vectors)

        return context_vectors
        

###test

#batch size, context length, number of embedded dimensions
batch = torch.randn(8, 1024, 800)
d_in = batch.shape[2] ## the input embedding size
d_out = 400 ## the output embedding size

print(batch.shape)

mha = MultiHeadAttention(d_in, d_out, batch.shape[1], 0, 2)

context_vecs = mha(batch)

print(context_vecs.shape)
