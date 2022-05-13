import torch
import torchvision
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import math 

class WineDataset(Dataset):
    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.num_samples = xy.shape[0]
    
    def __getitem__(self, index):
        # dataset[index]
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        return self.num_samples

dataset = WineDataset()

dataloader = DataLoader()
