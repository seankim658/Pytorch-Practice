# linear regression with gradint descent from scratch 

import torch
import numpy as np

# f = w * x: f = 2 * x 
x = np.array([1, 2, 3, 4], dtype = np.float32)
y = np.array([2, 4, 6, 8], dtype = np.float32)

# initial weight
w = 0.0

# model prediction 
def forward_pass(x):
    return w * x 

# loss: MSE
def loss_fn(y, y_pred):
    return ((y_pred - y)**2).mean()

# gradient 
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2x(w*x - y)
def gradient(x, y, y_preds):
    return np.dot(2 * x, y_preds - y).mean()

print(f'Initial predictions: f(5) = {forward_pass(5):.3f}')

# training
learning_rate = 0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_preds = forward_pass(x)
    # loss
    loss = loss_fn(y, y_preds)
    # gradients
    dw = gradient(x, y, y_preds)
    # update weight 
    w -= learning_rate * dw 
    if epoch % 2 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {loss:.4f}')

print(f'Final prediction: f(5) = {forward_pass(5):.3f}')