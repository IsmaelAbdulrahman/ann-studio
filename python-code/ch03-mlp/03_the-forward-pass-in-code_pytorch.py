# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 3: Multilayer networks & the forward pass
# Section: The forward pass in code
# Code example 3 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "mlp"
# ====================================================================

import torch, torch.nn as nn

mlp = nn.Sequential(
    nn.Linear(3, 4), nn.ReLU(),      # hidden layer
    nn.Linear(4, 2),                 # output layer (logits)
)
x = torch.randn(5, 3)                # a batch of 5 examples
logits = mlp(x)                      # forward pass: (5,3) -> (5,2)
print(logits.shape)
print(sum(p.numel() for p in mlp.parameters()), "parameters")
