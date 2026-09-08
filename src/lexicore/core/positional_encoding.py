import torch

class PositionalEncoding:
    def __init__(self,max_seq_len,dim_model):

        position = torch.arange(max_seq_len)

        i = torch.arange(dim_model)
        i = i.reshape(-1,2)

        pe = torch.zeros(max_seq_len,dim_model)

        for pair_num,index in enumerate(i):

            pe[:,index[0]] = torch.sin(position / 10000 ** (2 * pair_num / dim_model))
            pe[:,index[1]] = torch.cos(position / 10000 ** (2 * pair_num / dim_model))

        self.pe = pe


positional = PositionalEncoding(dim_model=8,max_seq_len=4)
print(positional.pe)