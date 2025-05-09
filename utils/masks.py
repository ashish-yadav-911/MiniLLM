# my_llm_project/utils/masks.py
import torch

def create_causal_mask(seq_len, device):
    """
    Creates a causal mask to prevent attention to future tokens.
    Shape: (seq_len, seq_len)
    Entries are 0 where attention is allowed, -inf where it's masked.
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1)
    return mask