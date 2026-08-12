# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 34: Case study: anomaly detection with autoencoders
# Section: In code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "cs-anomaly"
# ====================================================================

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# X_normal: (N, D) normal windows only (e.g. ECG5000 healthy beats, or sensor readings)
mu, sd = X_normal.mean(0), X_normal.std(0) + 1e-8    # normalize with NORMAL stats only
Xtr = (X_normal - mu) / sd
dl  = DataLoader(TensorDataset(Xtr), batch_size=64, shuffle=True)

class AE(nn.Module):
    def __init__(self, D, code=8):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(D, 32), nn.ReLU(), nn.Linear(32, code))
        self.dec = nn.Sequential(nn.Linear(code, 32), nn.ReLU(), nn.Linear(32, D))
    def forward(self, x):
        return self.dec(self.enc(x))                 # bottleneck: code << D

model = AE(Xtr.shape[1])
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)    # Adam: Chapter 9
for epoch in range(50):
    for (xb,) in dl:
        opt.zero_grad()
        loss = ((model(xb) - xb)**2).mean()          # reconstruction loss: Chapter 6
        loss.backward(); opt.step()

@torch.no_grad()
def score(X):                                        # anomaly score = per-row recon error
    Xn = (X - mu) / sd
    return ((model(Xn) - Xn)**2).mean(dim=1)

thr   = torch.quantile(score(X_val_normal), 0.99)    # threshold from NORMAL errors only
flags = score(X_test) > thr                          # above threshold  ->  anomaly
