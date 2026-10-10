import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads

        self.query = nn.Linear(dim_model, dim_model)
        self.key = nn.Linear(dim_model, dim_model)
        self.value = nn.Linear(dim_model, dim_model)
        self.out = nn.Linear(dim_model, dim_model)

    def forward(self, x, padding_mask=None):
        b, s, d = x.shape
        # shape -> [batch, seq_len, dim_model]

        # 1. Linear Projections & Reshape to [batch, num_heads, seq_len, head_dim]
        q = self.query(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).reshape(b, s, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. Scaled Dot-Product Attention -> shape: [batch, num_heads, seq_len, seq_len]
        attention_score = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 3. Apply Padding Mask safely
        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)

            attention_score = attention_score.masked_fill(padding_mask == True, float("-inf"))

        # 4. Softmax & Concatenation
        weights_attention = F.softmax(attention_score, dim=-1)

        w_v = torch.matmul(weights_attention, v)  # shape: [batch, num_heads, seq_len, head_dim]
        w_v = w_v.transpose(1, 2).reshape(b, s, self.dim_model)  # shape: [batch, seq_len, dim_model]

        out = self.out(w_v)

        return out