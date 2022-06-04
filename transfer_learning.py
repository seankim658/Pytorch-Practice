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

print('begun')

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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size = 4, shuffle = False) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def plot_images(input, title):
    '''imshow''' 
    # transpose image so it is [height x width x color channels]
    input = input.numpy().transpose(1, 2, 0)
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    plt.title(title)
    plt.show()

images, classes = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(images)
plot_images(out, title = [class_names[x] for x in classes])

def train(model, criterion, optimizer, scheduler, epochs = 25):
    since = time.time()

    best_model_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            correct = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward pass 
                # gradients on if in training phase 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward pass (only if in training phase)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step() 
                
                # track loss and correct predictions 
                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = correct.double() / dataset_sizes[phase]
    
            print(f'Epoch: {epoch}/{epochs - 1}, Phase: {phase}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(model.state_dict())

        print('----------')
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_weights)
    return model

### Two Approaches ### 

# First approach: load pretrained model, reset final fully connected layer, and retrain (finetune) all layers 

# load pre-trained model
model = models.resnet18(pretrained = True)
num_features = model.fc.in_features

# swap out model's fully connected layer for a new linear layer 
model.fc = nn.Linear(num_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.001)

# scheduler 
step_lr = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

model = train(model, criterion, optimizer, step_lr, epochs = 2)

# Second approach: load pretrained model and freeze all the network except the final layer
model2 = models.resnet18(pretrained = True)
for param in model2.parameters():
    param.requires_grad = False

num_features2 = model2.fc.in_features
model2.fc = nn.Linear(num_features2, 2)
model2.to(device)

criterion2 = nn.CrossEntropyLoss()
optimizer2 = optim.SGD(model2.parameters(), lr = 0.001)

# scheduler
step_lr2 = lr_scheduler.StepLR(optimizer2, step_size = 7, gamma = 0.1)

model2 = train(model2, criterion2, optimizer2, step_lr2, epochs = 2) 