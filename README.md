# My Simple Decoder-Only LLM Project

This project implements a basic Decoder-Only Transformer Language Model from scratch using PyTorch.
It's intended as an educational example to understand the core components of such models.

## Project Structure

- `config.py`: Global hyperparameters and settings.
- `data/`: Modules for data handling.
    - `tokenizer.py`: A simple character-level tokenizer.
    - `dataset.py`: PyTorch `Dataset` and `DataLoader` setup.
- `model/`: Core Transformer model components.
    - `transformer_blocks.py`: `MultiHeadSelfAttention`, `FeedForward`, `PositionalEncoding`.
    - `decoder.py`: `DecoderBlock`.
    - `llm.py`: The main `DecoderOnlyLLM` class.
- `utils/`: Utility functions.
    - `masks.py`: Causal mask creation.
- `train.py`: Script to train the LLM.
- `generate.py`: Script to generate text using a trained model.

## Setup

1.  Create a Python virtual environment (recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
2.  Install dependencies:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Or your CUDA/CPU version
    # No other external dependencies for this basic version
    ```

## Usage

### 1. Training the Model

The training script will:
- Create/load a character-level tokenizer (`char_tokenizer_vocab.json`).
- Prepare a dummy dataset.
- Train the `DecoderOnlyLLM`.
- Save the best model checkpoint to `trained_llm.pth` (or as specified in `config.py`).

To start training:
```bash
python -m my_llm_project.train# MiniLLM
