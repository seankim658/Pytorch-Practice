import torch 
import torch.nn as nn
import numpy as np

# from scratch 

def cross_entropy(actual, predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss 

# y is one hot encoded
y = np.array([1, 0, 0])

# y_pred has probabilities 
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])
l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)
print(f'L1 numpy: {l1:.4f}')
print(f'L2 numpy: {l2:.4f}')

# using pytorch 

loss = nn.CrossEntropyLoss()
# pytorch CEL implements softmax, and no need for one hot encoding
y = torch.tensor([0])
y_pred_good_torch = torch.tensor([[2.0, 1.0, 0.1]])
y_pred_bad_torch = torch.tensor([[0.5, 2.0, 0.3]])
l1_torch = loss(y_pred_good_torch, y)
l2_torch = loss(y_pred_bad_torch, y)
print(l1_torch.item())
print(l2_torch.item())

_, preds1 = torch.max(y_pred_good_torch, 1)
_, preds2 = torch.max(y_pred_bad_torch, 1)
print(preds1)
print(preds2)