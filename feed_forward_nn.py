import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms 
import matplotlib.pyplot as plt 
import numpy as np 

# device config 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters 
input_size = 28 * 28
hidden_size = 100
num_classes = 10
epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST data 
train_data = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.ToTensor(), download = True)
test_data = torchvision.datasets.MNIST(root = './data', train = False, transform = transforms.ToTensor(), download = True)

# data loaders 
train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = False)

# examples = iter(train_loader)
# samples, labels = examples.next()

# for i in range(9):
#     plt.subplot(3, 3, i + 1)
#     plt.imshow(samples[i][0], cmap = 'gray')
# plt.show()

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out 

# model 
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# loss and optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop 
n_total_steps = len(train_loader)
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        # reshape images 
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # forward pass 
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % 100 == 0:
            print(f'epoch {epoch + 1}/{epochs} [step {i + 1}/{n_total_steps}], loss = {loss.item():.4f}')

# test 
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        outputs = model(images)

        # value, index 
        _, preds = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (preds == labels).sum().item()
    
    acc = 100.0 * n_correct / n_samples
    print(f'accuracy = {acc:.2f}')