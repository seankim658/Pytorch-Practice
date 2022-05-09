# linear regression with gradint descent using autograd 

import torch

# f = w * x: f = 2 * x 
x = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

# initial weight
w = torch.tensor(0.0, dtype = torch.float32, requires_grad = True)

# model prediction 
def forward_pass(x):
    return w * x 

# loss: MSE
def loss_fn(y, y_pred):
    return ((y_pred - y)**2).mean()

print(f'Initial predictions: f(5) = {forward_pass(5):.3f}')

# training
learning_rate = 0.01
n_iters = 100

for epoch in range(n_iters):
    # prediction = forward pass
    y_preds = forward_pass(x)
    # loss
    loss = loss_fn(y, y_preds)
    # gradients = backward pass 
    loss.backward() # dl/dw 
    # update weight 
    with torch.no_grad():
        w -= learning_rate * w.grad
    # zero gradients 
    w.grad.zero_()
    if epoch % 10 == 0:
        print(f'epoch {epoch + 1}: w = {w:.3f}, loss = {loss:.4f}')

print(f'Final prediction: f(5) = {forward_pass(5):.3f}')