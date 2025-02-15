import torch
import sys

sys.path.append("../gpt")

from gpt import GPTModule
from helpers import *
from gpt_dataset import create_data_loader_v1
from torch.utils.data import Dataset, DataLoader

import os
from os import listdir
from os.path import isfile, join

import time

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
                                     

def train_model_simple(
        model, optimizer, device, num_epochs,
        eval_freq, eval_iter, start_context, tokenizer):

    training_losses = []
    validation_losses = []

    global_step = -1

    training_files_path = "/home/somindra/prog/gutenberg/data/raw/"
    training_files_list = "/home/somindra/prog/gutenberg/data/raw/gutenberg.list.english"

    training_files =[]
    training_file_names = []

    with open(training_files_list, "r", encoding="utf-8") as f:
        entry = f.readline()
        while entry:
            training_file_names.append(entry.rstrip('\n'))
            entry = f.readline()

    #this file is a long catalog of other files and it completely screws up the training
    training_file_names.remove('PG11800_raw.txt')

    print(training_file_names)
 
    for f in training_file_names:
        full_file_path = join(training_files_path,f)
        if isfile(full_file_path):
            training_files.append(full_file_path)

    print(f"training files list length: {len(training_files)}")
    avg_processing_time = 0

    for epoch in range(num_epochs):
        for filenum, file in enumerate(training_files[:100]):
            try:
                filenum = filenum + 1
                begin = time.time()
                with open(file, "r", encoding="utf-8") as f:
                    raw_text = f.read()

                train_ratio = 0.75 #75% of the text is for training, 25% for validation
                split_index = int(train_ratio * len(raw_text))

                train_text = raw_text[:split_index]
                validation_text = raw_text[split_index:]

                train_loader = create_data_loader_v1(
                    train_text,
                    batch_size=2,
                    max_length = GPT_CONFIG_124M["context_length"],
                    stride = GPT_CONFIG_124M["context_length"],
                    shuffle = True,
                    drop_last = True,
                    num_workers = 0)

                validation_loader = create_data_loader_v1(
                    validation_text,
                    batch_size=2,
                    max_length = GPT_CONFIG_124M["context_length"],
                    stride = GPT_CONFIG_124M["context_length"],
                    shuffle = True,
                    drop_last = True,
                    num_workers = 0)

                model.train()
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_batch, target_batch, model, device)
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(model, train_loader, validation_loader,
                                                              device, eval_iter)
                        training_losses.append(train_loss)
                        validation_losses.append(val_loss)

                        print(f"Step: {global_step} Training loss:{train_loss} Validation loss: {val_loss}")

                    if (global_step % 1000 == 0):
                        generate_and_print_sample(model, tokenizer, device, start_context)

                end = time.time()
                avg_processing_time = ((avg_processing_time * (filenum-1)) + (end-begin)) / filenum
                print(f"[{epoch+1}] Processed filenum, file, time, file size, avg proc time: {filenum},{os.path.basename(file)},{end-begin},{os.path.getsize(file)},{avg_processing_time}")

            except Exception as e:
                print(f"Caught an exception for file: {os.path.basename(file)}. Continuing.")
                continue
    
    return training_losses, validation_losses


###############################

import tiktoken
import json

GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,   # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}

print(f"GPT_CONFIG: {json.dumps(GPT_CONFIG_124M)}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
#device="cpu"

torch.manual_seed(123)
model = GPTModule(GPT_CONFIG_124M)
model.to(device)

lr = 4e-4
weight_decay=0.1
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)

print(f"Learning rate, Weight decay: {lr}, {weight_decay}")

tokenizer = tiktoken.get_encoding("gpt2")

begin = time.time()

num_epochs = 100
train_loss, val_loss = train_model_simple(
    model=model,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=100,
    eval_iter=5,
    start_context="While walking down the long road",
    tokenizer=tokenizer)

end = time.time()

print(f"Total training time: {end-begin}")

print(f"train_loss: {train_loss}")
print(f"val_loss: {val_loss}")

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, 
    "model_and_optimizer-cl1024.pth"
)

#While walking down the long road, he was notwithstanding the fact that he was a man of the world.  'I'm not sure,' he said, 'but I'm not sure that you are not a man of the world.'  'I'm

#While walking down the long road, he looked down upon the ground, and saw that the animal was not entered. He was not afraid of the lion’s approach, but he was not able to see that the lion was not at all surprised.


