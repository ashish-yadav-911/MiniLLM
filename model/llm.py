# my_llm_project/model/llm.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .transformer_blocks import PositionalEncoding
from .decoder import DecoderBlock
from ..utils.masks import create_causal_mask # Relative import from utils

class DecoderOnlyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, max_seq_len, dropout_rate):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len # Store for generation if needed

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len, dropout_rate)
        
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.output_norm = nn.LayerNorm(embed_dim) # Final layer norm before output
        self.output_layer = nn.Linear(embed_dim, vocab_size) # To predict next token
        self.dropout = nn.Dropout(dropout_rate)

        # Optional: Weight tying (share weights between embedding and output layer)
        # self.token_embedding.weight = self.output_layer.weight

        self.apply(self._init_weights) # Initialize weights

    def _init_weights(self, module):
        # Common weight initialization for Transformers
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


    def forward(self, input_ids, mask=None):
        # input_ids shape: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Token Embeddings
        x = self.token_embedding(input_ids) * math.sqrt(self.embed_dim) # Scale embeddings

        # 2. Positional Encoding
        x = self.positional_encoding(x)
        x = self.dropout(x) # Apply dropout after embeddings + pos_encoding

        # 3. Create Causal Mask if not provided (usually for training/generation)
        if mask is None:
            # Ensure mask is created on the same device as input_ids
            mask = create_causal_mask(seq_len, device=device)

        # 4. Pass through Decoder Blocks
        for block in self.decoder_blocks:
            x = block(x, mask)

        # 5. Final LayerNorm
        x = self.output_norm(x)

        # 6. Final Linear Layer to get logits
        logits = self.output_layer(x) # (batch_size, seq_len, vocab_size)
        return logits

    @torch.no_grad() # No gradients needed for generation
    def generate(self, start_tokens, max_new_tokens, temperature=1.0, top_k=None, device='cpu'):
        """
        Generates text sequence auto-regressively.
        start_tokens: tensor of shape (batch_size, initial_seq_len) or a list of token IDs for batch_size=1
        max_new_tokens: number of new tokens to generate
        temperature: for controlling randomness (higher -> more random)
        top_k: consider only top_k tokens for sampling
        device: 'cpu' or 'cuda'
        """
        self.eval() # Set model to evaluation mode
        
        if isinstance(start_tokens, list): # Handle single sequence input
            input_ids = torch.tensor([start_tokens], dtype=torch.long, device=device)
        else:
            input_ids = start_tokens.to(device)

        batch_size = input_ids.size(0)
        
        for _ in range(max_new_tokens):
            current_seq_len = input_ids.size(1)
            
            # Ensure input_ids do not exceed max_seq_len for positional encoding
            # If they do, take only the last max_seq_len tokens
            if current_seq_len > self.max_seq_len:
                input_ids_cond = input_ids[:, -self.max_seq_len:]
            else:
                input_ids_cond = input_ids
            
            # Create causal mask for the current (potentially truncated) sequence length
            causal_mask = create_causal_mask(input_ids_cond.size(1), device=device)
            
            logits = self(input_ids_cond, mask=causal_mask) # (batch_size, current_seq_len, vocab_size)
            
            next_token_logits = logits[:, -1, :] # (batch_size, vocab_size)

            if temperature <= 0: # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                    next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = F.softmax(next_token_logits, dim=-1) # (batch_size, vocab_size)
                next_token_id = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

            input_ids = torch.cat([input_ids, next_token_id], dim=1)
            
            # Optional: Stop if an EOS token is generated (assuming EOS_TOKEN_ID is defined)
            # if next_token_id.item() == config.EOS_TOKEN_ID:
            #     break
        
        if batch_size == 1:
            return input_ids.squeeze(0).tolist() # Return list of token IDs for single sequence
        return input_ids.tolist() # Return list of lists of token IDs for batch