# my_llm_project/train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import os

# from . import config # Project-level config
# from .model import DecoderOnlyLLM # Model from model/llm.py
# from .data.tokenizer import SimpleCharTokenizer
# from .data.dataset import create_data_loaders
import config # Assumes config.py is in the same directory
from model import DecoderOnlyLLM # Assumes model is a subdir
from data.tokenizer import SimpleCharTokenizer # Assumes data is a subdir
from data.dataset import create_data_loaders

def train_one_epoch(model, data_loader, loss_fn, optimizer, device, grad_clip_val, epoch_num):
    model.train()
    total_loss = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(data_loader):
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device) # These are also padded with PAD_TOKEN_ID

        optimizer.zero_grad()

        # The model's forward pass internally creates the causal mask
        logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

        # Reshape for CrossEntropyLoss:
        # Logits: (batch_size * seq_len, vocab_size)
        # Targets: (batch_size * seq_len)
        # The loss_fn will ignore PAD_TOKEN_ID in targets
        loss = loss_fn(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))

        loss.backward()
        if grad_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_val)
        optimizer.step()

        total_loss += loss.item()

        if batch_idx > 0 and batch_idx % config.LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            current_loss = loss.item()
            print(f"| Epoch {epoch_num:3d} | Batch {batch_idx:5d}/{len(data_loader):5d} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} "
                  f"| ms/batch {elapsed * 1000 / config.LOG_INTERVAL:5.2f} "
                  f"| loss {current_loss:5.2f} | ppl {math.exp(current_loss):8.2f}")
            start_time = time.time() # Reset timer for next log interval
            
    return total_loss / len(data_loader)

@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        target_ids = batch["target_ids"].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, config.VOCAB_SIZE), target_ids.view(-1))
        total_loss += loss.item()
    return total_loss / len(data_loader)

def main():
    print(f"Using device: {config.DEVICE}")

    # --- 1. Data Preparation & Tokenizer ---
    # Dummy corpus for this example
    raw_texts_train = [
        "hello world.", "this is a test sentence for training.", "llms are fun to build.",
        "transformers are powerful language models.", "the quick brown fox jumps over the lazy dog.",
        "another example to increase dataset size.", "character level tokenization is simple."
    ] * 20 # Multiply to make it a bit larger
    raw_texts_val = [
        "a new sentence for validation.", "evaluating the model performance.",
        "validation set helps to check overfitting."
    ] * 5

    # Initialize or load tokenizer
    tokenizer_path = "char_tokenizer_vocab.json"
    if os.path.exists(tokenizer_path):
        tokenizer = SimpleCharTokenizer(vocab_path=tokenizer_path)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:
        tokenizer = SimpleCharTokenizer(corpus=raw_texts_train + raw_texts_val) # Build from all text
        tokenizer.save_vocab(tokenizer_path)
        print(f"Built and saved tokenizer to {tokenizer_path}")

    # VOCAB_SIZE is now set in config by the tokenizer, PAD_TOKEN_ID too
    print(f"Tokenizer initialized. Vocab size: {config.VOCAB_SIZE}, PAD ID: {config.PAD_TOKEN_ID}")


    train_loader, val_loader = create_data_loaders(
        raw_texts_train, raw_texts_val, tokenizer,
        config.BATCH_SIZE, config.MAX_SEQ_LEN
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 2. Model Initialization ---
    model = DecoderOnlyLLM(
        vocab_size=config.VOCAB_SIZE, # From tokenizer
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        ff_dim=config.FF_DIM,
        max_seq_len=config.MAX_SEQ_LEN, # Max seq len for positional encoding
        dropout_rate=config.DROPOUT_RATE
    ).to(config.DEVICE)

    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # --- 3. Loss Function and Optimizer ---
    # Use ignore_index for padding tokens. config.PAD_TOKEN_ID is set by tokenizer.
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.PAD_TOKEN_ID)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95) # Example

    # --- 4. Training Loop ---
    best_val_loss = float('inf')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE, config.GRADIENT_CLIP_VAL, epoch)
        val_loss = evaluate(model, val_loader, loss_fn, config.DEVICE)
        
        epoch_duration = time.time() - epoch_start_time
        
        print("-" * 89)
        print(f"| end of epoch {epoch:3d} | time: {epoch_duration:5.2f}s | "
              f"train loss {train_loss:5.2f} | train ppl {math.exp(train_loss):8.2f} | "
              f"val loss {val_loss:5.2f} | val ppl {math.exp(val_loss):8.2f}")
        print("-" * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': {k: getattr(config, k) for k in dir(config) if not k.startswith('__')} # Save relevant config
            }, config.MODEL_SAVE_PATH)
            print(f"Saved best model to {config.MODEL_SAVE_PATH}")

        # if scheduler: scheduler.step() # Step the scheduler

    print("Training complete.")

if __name__ == "__main__":
    main()