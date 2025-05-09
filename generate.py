# my_llm_project/generate.py
import torch
import argparse

from . import config # Project-level config
from .model import DecoderOnlyLLM # Model
from .data.tokenizer import SimpleCharTokenizer # Tokenizer

def generate_text(model, tokenizer, start_string, max_new_tokens, temperature, top_k, device):
    model.eval() # Set model to evaluation mode

    # Encode the start string
    start_tokens = tokenizer.encode(start_string, add_special_tokens=True) # add BOS/EOS if tokenizer configured
    # Remove EOS if present, as we're generating from here. BOS is fine.
    if start_tokens[-1] == tokenizer.eos_id:
        start_tokens = start_tokens[:-1]

    input_ids = torch.tensor([start_tokens], dtype=torch.long, device=device)

    print(f"Generating from prompt (token IDs): {input_ids.tolist()}")
    print(f"Prompt (decoded): '{tokenizer.decode(input_ids.squeeze().tolist(), skip_special_tokens=False)}'") # Show BOS/EOS too

    generated_ids = model.generate(
        start_tokens=input_ids, # Pass as tensor
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        device=device
    )
    
    # generated_ids is a list of lists or list of token IDs if batch_size=1
    # For batch_size=1, model.generate returns a single list
    if isinstance(generated_ids, list) and (not generated_ids or isinstance(generated_ids[0], int)):
        decoded_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    else: # Should not happen with current generate for b_s=1
        print("Warning: Unexpected output format from model.generate()")
        decoded_text = "Error in decoding."


    return decoded_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained DecoderOnlyLLM.")
    parser.add_argument("--model_path", type=str, default=config.MODEL_SAVE_PATH,
                        help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--tokenizer_path", type=str, default="char_tokenizer_vocab.json",
                        help="Path to the tokenizer vocabulary file.")
    parser.add_argument("--prompt", type=str, default="hello",
                        help="Starting string for text generation.")
    parser.add_argument("--max_new_tokens", type=int, default=config.GENERATION_MAX_NEW_TOKENS,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=config.GENERATION_TEMPERATURE,
                        help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=config.GENERATION_TOP_K,
                        help="Top-k filtering for sampling (0 to disable).")
    parser.add_argument("--device", type=str, default=None, choices=['cpu', 'cuda'],
                        help="Device to use (cpu or cuda). Overrides config if set.")

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else config.DEVICE
    print(f"Using device: {device}")

    # --- 1. Load Tokenizer ---
    try:
        tokenizer = SimpleCharTokenizer(vocab_path=args.tokenizer_path)
        print(f"Loaded tokenizer from {args.tokenizer_path}. Vocab size: {config.VOCAB_SIZE}")
    except FileNotFoundError:
        print(f"Error: Tokenizer vocabulary file not found at {args.tokenizer_path}")
        print("Please train the model first (which creates the tokenizer vocab) or provide a valid path.")
        return

    # --- 2. Load Model ---
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        # If config was saved in checkpoint, you could use it to init model
        # For now, assume current config matches saved model structure
        # Or better, save model_config in checkpoint and load from there.
        # model_config = checkpoint.get('config', {}) # Get saved config if available

        model = DecoderOnlyLLM(
            vocab_size=config.VOCAB_SIZE, # Ensure this matches the tokenizer used for training
            embed_dim=config.EMBED_DIM,   # These should match the trained model
            num_heads=config.NUM_HEADS,
            num_layers=config.NUM_LAYERS,
            ff_dim=config.FF_DIM,
            max_seq_len=config.MAX_SEQ_LEN,
            dropout_rate=0.0 # No dropout during generation usually
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {args.model_path}")
    except FileNotFoundError:
        print(f"Error: Model checkpoint not found at {args.model_path}")
        print("Please train the model first or provide a valid path.")
        return
    except KeyError as e:
        print(f"Error loading model state_dict. Key missing: {e}. Ensure checkpoint is valid.")
        return


    # --- 3. Generate Text ---
    print(f"\nGenerating text with prompt: '{args.prompt}'")
    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        device
    )
    print("\n--- Generated Text ---")
    print(generated_text)
    print("----------------------")

if __name__ == "__main__":
    main()