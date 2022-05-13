import torch
import torch.nn as nn 
import numpy as np
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) prepare data 
bc = datasets.load_breast_cancer()
x, y = bc.data, bc.target

n_samples, n_features = x.shape

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1234)

# scale 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# tensors 
x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) model 
# f = wx + b, w/ sigmoid 
class LogisticRegression(nn.Module):
    def __init__(self, n_input):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input, 1)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))
    
model = LogisticRegression(n_features)

# 2) loss and optimizer 
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# 3) training loop
epochs = 100
for epoch in range(epochs):
    # forward pass
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)

    # backward pass
    loss.backward()

    # updates
    optimizer.step()

    # reset 
    optimizer.zero_grad()

    if (epoch + 1) % 10 == 0:
        print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')

with torch.no_grad():
    y_predicted = model(x_test)
    y_pred_labels = y_predicted.round()
    accuracy = y_pred_labels.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy = {accuracy:.4f}')