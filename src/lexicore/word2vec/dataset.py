from torch.utils.data import DataLoader,Dataset

class SkipGramDataset(Dataset):
    def __init__(self , pairs):
        self.pairs = pairs


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target,context = self.pairs[idx]

        return target,context


def create_dataloader(pairs, batch_size=32):

    dataset = SkipGramDataset(pairs)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader