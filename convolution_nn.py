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
epochs = 100
batch_size = 4
learning_rate = 0.001

# data set
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size, shuffle = False)

classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# cnn
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # W' = ((W - F + 2P) / S) + 1
        # ((32 - 5 + 2(0)) / 1) + 1 = 28
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 5)
        # ((28 - 2 + 2(0)) / 2) + 1 = 14
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # ((14 - 5 + 2(0)) / 1) + 1 = 10
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        # after second pooling operation:
        # ((10 - 2 + 2(0)) / 2) + 1 = 5
        self.fc1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
        self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        self.fc3 = nn.Linear(in_features = 84, out_features = 10) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten before passing to fully connected layers 
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = CNN().to(device)

error = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

n_steps = len(train_loader)

# training
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

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
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        _, preds = torch.max(outputs, 1)
        samples += labels.size(0)
        correct += (preds == labels).sum()
        
        for i in range(batch_size):
            label = labels[i]
            pred = preds[i]
            if label == pred:
                class_correct[label] += 1
            class_sample[label] += 1

    accuracy = 100.0 * correct / samples 
    print(f'Accuracy: {accuracy:.2f}%')

    for i in range(10):
        class_accuracy = 100.0 * class_correct[i] / class_sample[i]
