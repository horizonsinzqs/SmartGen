import torch
from torch.utils.data import Dataset
import pickle
import numpy

class LogDataSet(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = pickle.load(open(self.file_path, 'rb'))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        # Return: [:-1, [0,1,2,4]], last row [0], last row [1], last row [4]
        return self.data[idx][0 : -1, [0, 1, 2, 4]], self.data[idx][-1, [0]], self.data[idx][-1, [1]], self.data[idx][-1, [4]]

class RoutineDataSet(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataList = []
        with open(file_path) as f:
            self.dataList = f.read().split('\n')

    def __len__(self):
        return len(self.dataList)

    def __getitem__(self, idx):
        # Return tensor of integer list
        return torch.tensor(list(map(int, self.dataList[idx].split())))