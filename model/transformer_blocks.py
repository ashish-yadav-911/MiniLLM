# my_llm_project/model/transformer_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # x shape: (batch_size, seq_len, embed_dim)
        batch_size, seq_len, _ = x.shape

        # 1. Linear projections
        q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Split into multiple heads
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            # Mask needs to be broadcastable: (1, 1, seq_len, seq_len) for causal
            # or (batch_size, 1, 1, seq_len) for padding mask (not used in decoder-only causal)
            scores = scores + mask # Mask is (seq_len, seq_len) -> auto-broadcasts

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, v)

        # 4. Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_proj(context)
        return output # We don't return attention_weights by default, but can if needed

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout_rate):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU() # Common in Transformers

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len, dropout_rate):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

        pe = torch.zeros(max_seq_len, embed_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension: (1, max_seq_len, embed_dim)

        self.register_buffer('pe', pe) # Not a parameter, but part of the state

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_dim)
        # self.pe is (1, max_seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :] # Select appropriate part of pe
        return self.dropout(x)