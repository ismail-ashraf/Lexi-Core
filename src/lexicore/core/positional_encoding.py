import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dim_model):
        super().__init__()

        position = torch.arange(max_seq_len)
        i = torch.arange(dim_model).reshape(-1, 2)

        pe = torch.zeros(max_seq_len, dim_model)

        for pair_num, index in enumerate(i):
            pe[:, index[0]] = torch.sin(position / 10000 ** (2 * pair_num / dim_model))
            pe[:, index[1]] = torch.cos(position / 10000 ** (2 * pair_num / dim_model))

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]

        return x