# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Activations and losses
# Code example 6 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import torch.nn.functional as F

z = torch.tensor([-1.0, 0.0, 2.0])
print("relu   :", F.relu(z))
print("sigmoid:", torch.sigmoid(z))
print("softmax:", F.softmax(z, dim=0))    # sums to 1

# regression: mean squared error
pred = torch.tensor([2.5, 0.0, 2.0])
tgt  = torch.tensor([3.0, -0.5, 2.0])
print("MSE:", F.mse_loss(pred, tgt).item())

# classification: raw logits + integer labels (NOT one-hot)
logits = torch.tensor([[2.0, 0.5, 0.1]])   # one sample, 3 classes
label  = torch.tensor([0])                 # the true class index
print("cross-entropy:", F.cross_entropy(logits, label).item())
