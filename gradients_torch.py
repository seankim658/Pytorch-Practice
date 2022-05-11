# linear regression with pytorch

# 1) Design model (input, output size, forward pass)
# 2) Construct loss and optimizer 
# 3) Training loop
#   - forward pass: compute prediciton 
#   - backward pass: gradients 
#   - update weights 

from pickletools import optimize
from numpy import float32
import torch
import torch.nn as nn 

# f = w * x: f = 2 * x 
x = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)

n_samples, n_features = x.shape
input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

print(f'Initial predictions: f(5) = {model(torch.tensor([5], dtype = torch.float32)).item():.3f}')

# training
learning_rate = 0.01
n_iters = 150

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_preds = model(x)
    # loss
    l = loss(y, y_preds)
    # gradients = backward pass 
    l.backward() # dl/dw 
    # update weight 
    optimizer.step()
    # zero gradients 
    optimizer.zero_grad()
    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {model.weight.item():.3f}, loss = {l:.4f}')
        # for param in model.parameters():
        #     print(param)
        print('---------------------------')

print(f'Final prediction: f(5) = {model(torch.tensor([5], dtype = torch.float32)).item():.3f}')