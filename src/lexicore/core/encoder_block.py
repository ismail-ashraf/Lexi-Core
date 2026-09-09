import torch.nn as nn
from .multi_head_attentions import MultiHeadAttention


class EncoderBlock(nn.Module):
    def __init__(self,dim_model,num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(dim_model,num_heads)
        self.norm1 = nn.LayerNorm(dim_model)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model,2048),
            nn.ReLU(),
            nn.Linear(2048,dim_model)
        )
        self.norm2 = nn.LayerNorm(dim_model)

    def forward(self,x):
        attention_out = self.attention(x)
        print("MHA Output Shape:",attention_out.shape)
        x = x + attention_out
        norm1 = self.norm1(x)
        ffn = self.ffn(norm1)
        x = x + ffn
        norm2 = self.norm2(x)

        return norm2


class Encoder(nn.Module):
    def __init__(self,num_layers , dim_model,num_heads):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderBlock(dim_model,num_heads)
            for _ in range(num_layers)
        ])

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            """
            for example:: -> X -> block 1 -> X1 -> block 2 -> X2 -> block 3 -> X3
            """
        return x