from lexicore.core.positional_encoding import PositionalEncoding
from lexicore.core.encoder_block import Encoder
import torch.nn as nn
import torch

class TransformerEncoderModel(nn.Module):
    def __init__(self,max_seq_len,dim_model,num_heads,num_layers,vocab_size,pad_id):
        super().__init__()

        self.positional_encoder = PositionalEncoding(max_seq_len=max_seq_len , dim_model=dim_model)
        self.encoder_blocks = Encoder(num_layers , dim_model,num_heads,pad_id)
        self.classifier = nn.Linear(dim_model,3)

        self.embedding = nn.Embedding(
            vocab_size,
            dim_model
        )
        self.pad_id = pad_id

    def forward(self,x):
        padding_mask = (x == self.pad_id)

        embedding = self.embedding(x)

        positional_encoder = self.positional_encoder(embedding)

        encoder_block = self.encoder_blocks(positional_encoder,padding_mask)

        mask = (~padding_mask).unsqueeze(-1) #type:ignore

        masked_encoder = encoder_block * mask

        sum_embeddings = masked_encoder.sum(dim=1)

        token_count = mask.sum(dim=1)

        mean = sum_embeddings / token_count
        # shape -> [batch,dim_model]

        classifier = self.classifier(mean)

        print("classifier logits",classifier)
        # shape -> [batch,3 classifier output]

        return classifier

model = TransformerEncoderModel(
    max_seq_len=10,
    dim_model=128,
    num_heads=8,
    num_layers=3,
    vocab_size=1000,
    pad_id=1
)

x = torch.randn(1,10,128)

out = model(x)

print(out.shape)