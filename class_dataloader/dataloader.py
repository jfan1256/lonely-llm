from torch.utils.data import Dataset

from class_dataloader.utils import preprocess_text

# Pretrain Dataloader
class Pretrain(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        narrative = preprocess_text(self.data.iloc[index]['narrative'])
        inputs = self.tokenizer(narrative, truncation=True, padding='max_length', max_length=512)
        return inputs

# Train Dataloader
class Train(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        narrative = preprocess_text(self.data.iloc[index]['narrative'])
        label = self.data.iloc[index]['label']
        reason = self.data.iloc[index]['reason']
        return index, narrative, label, reason