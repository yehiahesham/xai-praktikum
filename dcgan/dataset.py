import torch.nn as nn
import torch

class Dataset(nn.Module):
    def __init__(self, data, targets):
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        current_sample = self.data[index, :]
        current_target = self.targets[index]
        return {
            "sample": torch.tensor(current_sample, dtype=torch.float),
            "target": torch.tensor(current_target, dtype=torch.long),
        }

