# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Putting it together: a complete run
# Code example 12 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. toy data: two Gaussian blobs -> a 2-class problem
n = 400
X = torch.randn(n, 2)
X[:n // 2] += 2.0                       # push class 0 up and to the right
y = torch.zeros(n, dtype=torch.long)    # int64 labels for CrossEntropyLoss
y[n // 2:] = 1
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

# 2. model, loss, optimizer
model = nn.Sequential(
    nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 2)
).to(device)
loss_fn = nn.CrossEntropyLoss()         # expects raw logits + int64 labels
opt = torch.optim.Adam(model.parameters(), lr=1e-2)

# 3. train: the five steps, every batch, every epoch
for epoch in range(20):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()                 # 1. clear
        logits = model(xb)              # 2. forward
        loss = loss_fn(logits, yb)      # 3. loss
        loss.backward()                 # 4. backward (autograd)
        opt.step()                      # 5. update

# 4. evaluate: eval mode + no gradient tracking
model.eval()
with torch.no_grad():
    pred = model(X.to(device)).argmax(dim=1).cpu()
acc = (pred == y).float().mean().item()
print(f"train accuracy: {acc:.3f}")     # ~0.99 on these well-separated blobs
