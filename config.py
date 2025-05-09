# my_llm_project/config.py

import torch

# --- Data & Tokenizer Configuration ---
# For a real project, you'd train a tokenizer (e.g., BPE) on your corpus
# and save it. Then VOCAB_SIZE would be determined by the tokenizer.
# For this example, we'll use a simple character-level setup.
VOCAB_SIZE = None  # Will be set by the tokenizer
MAX_SEQ_LEN = 64   # Maximum sequence length the model can handle during training and generation
PAD_TOKEN_ID = 0   # Placeholder, will be set by tokenizer
BOS_TOKEN_ID = 1   # Placeholder, will be set by tokenizer
EOS_TOKEN_ID = 2   # Placeholder, will be set by tokenizer
UNK_TOKEN_ID = 3   # Placeholder, will be set by tokenizer

# --- Model Configuration ---
EMBED_DIM = 128       # Dimension of token embeddings and model hidden states
NUM_HEADS = 4         # Number of attention heads
NUM_LAYERS = 3        # Number of decoder blocks
FF_DIM = EMBED_DIM * 4 # Hidden dimension of the feed-forward network (common practice)
DROPOUT_RATE = 0.1

# Ensure embed_dim is divisible by num_heads
assert EMBED_DIM % NUM_HEADS == 0, "EMBED_DIM must be divisible by NUM_HEADS"

# --- Training Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 32 # Adjust based on your GPU memory
NUM_EPOCHS = 20 # For a real model, much more, and depends on dataset size
GRADIENT_CLIP_VAL = 1.0
MODEL_SAVE_PATH = "trained_llm.pth"
LOG_INTERVAL = 10 # Log training progress every N batches

# --- Generation Configuration ---
GENERATION_MAX_NEW_TOKENS = 100
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 50