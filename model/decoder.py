# my_llm_project/model/decoder.py
import torch.nn as nn
from .transformer_blocks import MultiHeadSelfAttention, FeedForward

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate)
        self.feed_forward = FeedForward(embed_dim, ff_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_rate) # Dropout for residual connections

    def forward(self, x, mask):
        # x shape: (batch_size, seq_len, embed_dim)
        # mask shape: (seq_len, seq_len) for causal mask

        # 1. Self-Attention with Pre-LayerNorm and Residual
        # Pre-Norm: Normalize input before passing to sub-layer
        normed_x = self.norm1(x)
        attn_output = self.self_attn(normed_x, mask=mask)
        x = x + self.dropout(attn_output) # Residual connection

        # 2. Feed-Forward with Pre-LayerNorm and Residual
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output) # Residual connection
        return x