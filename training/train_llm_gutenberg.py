import torch
import sys

sys.path.append("../gpt")

from gpt import GPTModule
from helpers import *
from gpt_dataset import create_data_loader_v1
from torch.utils.data import Dataset, DataLoader

#calculates the cross entropy loss in batch
def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),
                                             target_batch.flatten())

    return loss

def calc_loss_loader(loader, model, device, num_batches=None):
    total_loss = 0.
    if len(loader) == 0:
        return float("nan")
    elif num_batches is None: #evaluate the entire loader; all the batches it has
        num_batches = len(loader)
    else:
        num_batches = min(len(loader), num_batches)

    for i, (input_batch, target_batch) in enumerate(loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss/num_batches
        
                     
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, eval_iter)

    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    ##start_context is a just a string, the context that you pass

    model.eval()
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(model=model,
                                         idx=encoded,
                                         max_new_tokens=50,
                                         context_size=1024)

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n"," "))
    model.train()
                                     

def train_model_simple(model, train_loader, val_loader,
                       optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    training_losses = []
    validation_losses = []

    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader,
                                                  device, eval_iter)

                training_losses.append(train_loss)
                validation_losses.append(val_loss)

                print(f"Epoch: {epoch} Step: {global_step} Training loss:{train_loss} Validation loss: {val_loss}")


        generate_and_print_sample(model, tokenizer, device, start_context)

    return training_losses, validation_losses


###############################

import tiktoken

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device="cpu"

with open("/home/somindra/prog/gutenberg/chunks/chunk.1", "r", encoding="utf-8") as f:
    raw_text = f.read()

train_ratio = 0.75 #90% of the text is for training, 10% for validation
split_index = int(train_ratio * len(raw_text))
                  
train_text = raw_text[:split_index]
validation_text = raw_text[split_index:]

print(f"validation_text length = {len(validation_text)}")

train_loader = create_data_loader_v1(train_text,
                                     batch_size=2,
                                     max_length = GPT_CONFIG_124M["context_length"],
                                     stride = GPT_CONFIG_124M["context_length"],
                                     shuffle = True,
                                     drop_last = True,
                                     num_workers = 0)

print(f"train_loader length: {len(train_loader)}")

validation_loader = create_data_loader_v1(validation_text,
                                          batch_size=2,
                                          max_length = GPT_CONFIG_124M["context_length"],
                                          stride = GPT_CONFIG_124M["context_length"],
                                          shuffle = True,
                                          drop_last = True,
                                          num_workers = 0)

print(f"validation_loader length = {len(validation_loader)}")

torch.manual_seed(123)
model = GPTModule(GPT_CONFIG_124M)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(),
                              lr=0.0004,
                              weight_decay=0.1)

num_epochs = 10

tokenizer = tiktoken.get_encoding("gpt2")

train_loss, val_loss = train_model_simple(
    model=model, train_loader=train_loader, val_loader=validation_loader, optimizer=optimizer,
    device=device, num_epochs=num_epochs, eval_freq=5, eval_iter=5, start_context="Every effort moves you",
    tokenizer=tokenizer)

print(train_loss)

torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer-cl1024.pth"
)
