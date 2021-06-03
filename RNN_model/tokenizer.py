from torch.utils.data import DataLoader, Dataset
import torch 

class ToxicityDataset(Dataset):
    def __init__(self, text):
        self.text = text
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        return {
            'text': torch.tensor(self.text[item], dtype=torch.long),
        }