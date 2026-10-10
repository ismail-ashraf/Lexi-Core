import torch
import torch.nn as nn
from lexicore.core.positional_encoding import PositionalEncoding
from lexicore.core.encoder_block import Encoder


class TransformerEncoderModel(nn.Module):
    def __init__(self, max_seq_len, dim_model, num_heads, num_layers, vocab_size, pad_id):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.pad_id = pad_id

        self.embedding = nn.Embedding(
            vocab_size,
            dim_model
        )
        self.positional_encoder = PositionalEncoding(max_seq_len=max_seq_len, dim_model=dim_model)
        self.encoder_blocks = Encoder(num_layers, dim_model, num_heads, pad_id)
        self.classifier = nn.Linear(dim_model, 3)

    def forward(self, x):
        if x.size(1) > self.max_seq_len:
            x = x[:, :self.max_seq_len]

        padding_mask = (x == self.pad_id)

        embedding = self.embedding(x.long())
        positional_encoder = self.positional_encoder(embedding)

        encoder_block = self.encoder_blocks(positional_encoder, padding_mask)

        mask = (~padding_mask).unsqueeze(-1)  # shape: [batch, seq_len, 1]
        masked_encoder = encoder_block * mask

        sum_embeddings = masked_encoder.sum(dim=1)
        token_count = mask.sum(dim=1).clamp(min=1e-9)

        mean = sum_embeddings / token_count  # shape: [batch, dim_model]

        classifier = self.classifier(mean)  # shape: [batch, 3]

        return classifier
