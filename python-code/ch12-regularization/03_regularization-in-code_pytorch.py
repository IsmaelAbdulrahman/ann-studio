# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 12: Regularization & generalization
# Section: Regularization in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "regularization"
# ====================================================================

import torch, torch.nn as nn

net = nn.Sequential(
    nn.Linear(20, 128), nn.ReLU(),
    nn.Dropout(0.5),                       # inverted dropout, active in train() only
    nn.Linear(128, 1))

# L2 weight decay lives in the optimizer; AdamW decouples it cleanly (Chapter 9)
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)

net.train()                               # dropout ON, masks resampled each step
for xb, yb in loader:                     # your DataLoader of (features, target)
    opt.zero_grad()
    loss = nn.functional.mse_loss(net(xb), yb)
    loss.backward(); opt.step()

net.eval()                                # dropout OFF at test time — full network used
