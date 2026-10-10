import torch.nn as nn
from .multi_head_attentions import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self, dim_model, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(dim_model, num_heads)
        self.norm1 = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, 2048),
            nn.ReLU(),
            nn.Linear(2048, dim_model)
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self, x, padding_mask=None):
        attention_out = self.attention(x, padding_mask)
        x = x + attention_out
        norm1 = self.norm1(x)
        ffn = self.ffn(norm1)
        x = x + ffn
        norm2 = self.norm2(x)

        return norm2


class Encoder(nn.Module):
    def __init__(self, num_layers, dim_model, num_heads, pad_id):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(dim_model, num_heads)
            for _ in range(num_layers)
        ])
        self.pad_id = pad_id

    def forward(self, x, padding_mask=None):
        for layer in self.layers:
            x = layer(x, padding_mask)

        return x