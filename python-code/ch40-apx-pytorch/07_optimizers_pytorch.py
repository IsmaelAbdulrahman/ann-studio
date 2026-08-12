# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Optimizers
# Code example 7 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import torch.nn as nn

model = nn.Linear(10, 1)

# SGD with momentum and L2 weight decay
opt = torch.optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=1e-4)

# Adam: the common default; adaptive per-parameter step sizes
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# parameter groups: a different lr for different tensors
opt = torch.optim.Adam([
    {"params": model.weight, "lr": 1e-3},
    {"params": model.bias,   "lr": 1e-2},
])
print(opt)
