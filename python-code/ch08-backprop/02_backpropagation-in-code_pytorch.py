# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 8: Backpropagation
# Section: Backpropagation in code
# Code example 2 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "backprop"
# ====================================================================

import torch, torch.nn as nn

x = torch.tensor([[0.05, 0.10]])                 # batch of one
y = torch.tensor([[0.70]])
net = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 1))
with torch.no_grad():                            # plant the chapter's weights
    net[0].weight.copy_(torch.tensor([[0.15, 0.20], [0.25, 0.30]]))  # = W1^T
    net[0].bias.copy_(torch.tensor([0.35, 0.35]))
    net[2].weight.copy_(torch.tensor([[0.40, 0.50]]))                # = W2^T
    net[2].bias.copy_(torch.tensor([0.60]))

yhat = net(x)
loss = 0.5 * (yhat - y).pow(2).sum()             # MSE, batch of one
loss.backward()                                  # one call runs all of backprop
print(yhat.item(), loss.item())                  # 1.1358 , 0.0949
print(net[2].weight.grad)                         # -> [[0.2585, 0.2601]]
print(net[0].weight.grad)                         # hidden-layer gradients (W1^T layout)
