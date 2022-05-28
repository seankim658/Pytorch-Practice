from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optim 
from torch.optim import lr_scheduler
import numpy as np
import torchvision 
from torchvision import datasets, models, transforms 
import matplotlib.pyplot as plt
import time 
import os 
import copy 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

data_root = 'data/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle = True) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def plot_images(input, title):
    '''imshow''' 
    # transpose image so it is [height x width x color channels]
    input = input.numpy().transpose(1, 2, 0)
    input = std * input + mean
    plt.imshow(input)
    plt.title(title)
    plt.show()

images, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(images)
plot_images(out, title = [class_names[x] for x in classes])