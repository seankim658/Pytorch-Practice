import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt
import numpy as np 

# gpu; if available 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters 
epochs = 4
batch_size = 4
learning_rate = 0.001

# data set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
])
train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# cnn
class CNN(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

model = CNN().to(device)

error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_steps = len(train_loader)

# training
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        lables = labels.to(device)

        # forward pass 
        outputs = model(images)
        loss = error(outputs, labels)

        # backward 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 2000 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}] | Step [{i + 1}/{n_steps}] | Loss: {loss.item():.4f}')

# testing 
with torch.no_grad():
    correct = 0
    samples = 0
    class_correct = [0 for i in range(10)]
    class_sample = [0 for i in range(10)]
    for images, labels in test_loader:
        print(labels)
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        samples += labels.size(0)
        correct += (preds == labels).sum()
        
        for i in range(batch_size):
            label = labels[i]
            pred = preds[i]

    accuracy = 100.0 * np.correct / samples 
    print(f'Accuracy: {accuracy:.2f}%')

    for i in range(10):
        class_accuracy = 100.0 * class_correct[i] / class_sample[i]
