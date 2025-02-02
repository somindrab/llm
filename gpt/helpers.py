import torch


## asks a model to generate text given some input
def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        trimmed_idx = idx[:, -context_size:] #use only the last context_size as context if len(idx) > context_size

        with torch.no_grad():
            logits = model(trimmed_idx)

        logits = logits[:, -1, :] #we want only the last token
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

## uses a user supplied tokenizer to convert text into tokens
## returns a tensor of tokens
def text_to_token_ids(text, tokenizer):
    token_ids = tokenizer.encode(text,
                                 allowed_special={'|endoftext|'})

    #unsqueezes the first dimension, i.e., adds a batch dimension
    encoded_tensor = torch.tensor(token_ids).unsqueeze(0)

    return encoded_tensor

## uses a user supplied tokenizer to decode the tokenids
## and return the text as a list
def token_ids_to_text(tokenids, tokenizer):
 
    #remember that the first dimension will be a batch
    #so we squeeze to get rid of it
    #and we want to return a list, not a tensor

    flat_tokens = tokenids.squeeze(0)
    decoded_text = tokenizer.decode(flat_tokens.tolist())
    return decoded_text


