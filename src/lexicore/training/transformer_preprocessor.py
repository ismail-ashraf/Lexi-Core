import torch
from torch.utils.data import Dataset


class TransformerPreprocessor:
    def __init__(self,vocabulary,tokenizer,max_seq_len):
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.max_seq_len = max_seq_len

    def encode_text(self,text):
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]

        if len(tokens) < self.max_seq_len:
            tokens += ['<PAD>'] * (self.max_seq_len - len(tokens))

        token_ids = self.vocabulary.encode(tokens)

        return token_ids

    def encode_dataset(self,texts):
        encoded = []

        for text in texts:
            encoded.append(self.encode_text(text))

        encoded = torch.tensor(
            encoded,
            dtype=torch.long
        )

        return encoded


class TransformerDataset(Dataset):
    def __init__(self,X,y):
        self.X = X
        self.y = torch.tensor(y,dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return self.X[item] , self.y[item]