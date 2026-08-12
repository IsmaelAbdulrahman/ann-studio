# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: The losses in code
# Code example 4 of 8 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import torch
import torch.nn as nn

# regression: mean squared error (PyTorch omits the 1/2 factor)
yhat = torch.tensor([2.5, -0.2, 2.0, 1.5])
y    = torch.tensor([3.0, -1.0, 2.0, 0.5])
print(nn.MSELoss()(yhat, y))            # 0.4725  == 2 * 0.2362

# classification: cross-entropy on RAW logits + integer labels
logits = torch.tensor([[1.0, 2.0, 0.5]])   # shape (batch=1, classes=3)
target = torch.tensor([1])                  # true class index (not one-hot)
print(nn.CrossEntropyLoss()(logits, target))   # 0.4644, fuses log-softmax + NLL

# for a single sigmoid output, use BCEWithLogitsLoss (also fused / stable)
z = torch.tensor([0.3]); t = torch.tensor([1.0])
print(nn.BCEWithLogitsLoss()(z, t))
