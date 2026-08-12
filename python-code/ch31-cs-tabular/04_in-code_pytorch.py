# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: In code
# Code example 4 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# X_std: (N, d) standardized features; y_lbl: (N,) labels in {0., 1.}
X = torch.tensor(X_std, dtype=torch.float32)
y = torch.tensor(y_lbl, dtype=torch.float32)
ds = TensorDataset(X, y)
n_val = int(0.2 * len(ds))
train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl   = DataLoader(val_ds,   batch_size=256)

net = nn.Sequential(
    nn.Linear(X.shape[1], 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 16),         nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(16, 1))                                  # a single logit

pos_weight = torch.tensor([(y == 0).sum() / (y == 1).sum()])   # up-weight rare churners
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

best, wait, patience = float("inf"), 0, 5
for epoch in range(100):
    net.train()
    for xb, yb in train_dl:
        opt.zero_grad()
        loss_fn(net(xb).squeeze(1), yb).backward()
        opt.step()
    net.eval(); vloss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            vloss += loss_fn(net(xb).squeeze(1), yb).item() * len(xb)
    vloss /= len(val_ds)
    if vloss < best - 1e-4:
        best, wait = vloss, 0
        torch.save(net.state_dict(), "best.pt")        # checkpoint the best model
    else:
        wait += 1
        if wait >= patience:
            print(f"early stop at epoch {epoch}, best val loss {best:.4f}")
            break
