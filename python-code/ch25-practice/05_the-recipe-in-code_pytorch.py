# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: The recipe in code
# Code example 5 of 8 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def make_loader(X, y, bs=64, shuffle=False):
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle)
# train_loader = make_loader(Xtr, ytr, shuffle=True)   # build from your splits
# val_loader   = make_loader(Xva, yva)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)   # Adam @ 1e-3: strong default
lossf = nn.CrossEntropyLoss()

def run_epoch(loader, train):
    model.train() if train else model.eval()
    tot = correct = 0; loss_sum = 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            logits = model(xb); loss = lossf(logits, yb)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(xb)
            correct  += (logits.argmax(1) == yb).sum().item(); tot += len(xb)
    return loss_sum / tot, correct / tot

best, wait, patience = 1e9, 0, 5
for epoch in range(100):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    print(f"epoch {epoch}: train {tr_loss:.3f}/{tr_acc:.3f}  val {va_loss:.3f}/{va_acc:.3f}")
    if va_loss < best:                       # early stopping on validation loss
        best, wait = va_loss, 0
        torch.save(model.state_dict(), "best.pt")
    else:
        wait += 1
        if wait >= patience:
            break
