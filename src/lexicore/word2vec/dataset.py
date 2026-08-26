from torch.utils.data import DataLoader,Dataset
from lexicore.word2vec.preprocessor_component import pairs

class SkipGramDataset(Dataset):
    def __init__(self , pairs):
        self.pairs = pairs


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        target,context = self.pairs[idx]

        return target,context



dataset = SkipGramDataset(pairs)

dataloader = DataLoader(batch_size=32 , dataset=dataset,shuffle=True)