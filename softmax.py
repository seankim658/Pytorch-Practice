import torch 
import torch.nn as nn
import numpy as np 

# from scratch 
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis = 0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy:', outputs)

# using pytorch 
x_tensor = torch.tensor([2.0, 1.0, 0.1])
outputs_torch = torch.softmax(x_tensor, dim = 0)
print('softmax torch:', outputs_torch)

