import torch
import pandas as pd

class ChapmanTrainDataset(torch.utils.data.Dataset):
    def __init__(self, data_train):  
        self.data_train = data_train 
        self.data = data_train.iloc[:, 1:].values  # Exclude first column (index)
        self.labels = data_train.iloc[:, 0].values  # First column is labels

    def __len__(self):
        return self.data_train.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label

class ChapmanTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_test):   
        self.data_test = data_test
        self.data = self.data_test.iloc[:, 1:].values  # Exclude first column (index)
        self.labels = self.data_test.iloc[:, 0].values  # First column is labels

    def __len__(self):
        return self.data_test.shape[0]

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return sample, label