from lexicore.core.positional_encoding import PositionalEncoding
from lexicore.core.encoder_block import Encoder
import torch.nn as nn
import torch

class TransformerEncoderModel(nn.Module):
    def __init__(self,max_seq_len,dim_model,num_heads,num_layers):
        super().__init__()

        self.positional_encoder = PositionalEncoding(max_seq_len=max_seq_len , dim_model=dim_model)
        self.encoder_blocks = Encoder(num_layers , dim_model,num_heads)
        self.classifier = nn.Linear(dim_model,3)

    def forward(self,x):
        positional_encoder = self.positional_encoder(x)
        encoder_block = self.encoder_blocks(positional_encoder)
        mean = encoder_block.mean(dim=1)
        # shape -> [batch,dim_model]
        classifier = self.classifier(mean)
        print("classifier logits",classifier)
        # shape -> [batch,3 classifier output]

        return classifier

model = TransformerEncoderModel(
    max_seq_len=10,
    dim_model=128,
    num_heads=8,
    num_layers=3
)

x = torch.randn(1,10,128)

out = model(x)

print(out.shape)