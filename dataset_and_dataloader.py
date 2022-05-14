import torch
import torchvision
from torch.utils.data import Dataset, DataLoader 
import numpy as np
import math 

class WineDataset(Dataset):
    def __init__(self, transform = None):
        # data loading
        xy = np.loadtxt('./data/wine.csv', delimiter = ',', dtype = np.float32, skiprows = 1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]] # n_samples, 1
        self.num_samples = xy.shape[0]
        self.transform = transform
    
    def __getitem__(self, index):
        # dataset[index]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # len(dataset)
        return self.num_samples

class ToTensor:
    def __call__(self, sample):
        inputs, labels = sample
        return torch.from_numpy(inputs), torch.from_numpy(labels)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor 
    
    def __call__(self, sample):
        inputs, labels = sample 
        inputs *= self.factor
        return inputs, labels

dataset = WineDataset(transform = ToTensor())

dataloader = DataLoader(dataset = dataset, batch_size = 4, shuffle = True)

# training loop
epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples / 4)

for epoch in range(epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward pass
        if (i + 1) % 5 == 0:
            print(f'epoch: {epoch + 1}/{epochs}, step {i + 1}/{n_iterations}, inputs {inputs.shape}')
