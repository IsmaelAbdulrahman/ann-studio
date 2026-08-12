# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 33: Case study: time-series forecasting
# Section: In code
# Code example 4 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "cs-sequence"
# ====================================================================

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def make_windows(series, L, H=1):             # series: 1-D tensor already split in TIME
    xs = [series[i:i+L]        for i in range(len(series)-L-H+1)]
    ys = [series[i+L:i+L+H]    for i in range(len(series)-L-H+1)]
    X = torch.stack(xs).unsqueeze(-1)         # (N, L, 1)  -> one feature per step
    return X, torch.stack(ys)                 # y: (N, H)

L, H = 24, 1
mu, sd = train.mean(), train.std()            # normalize with TRAIN stats ONLY
Xtr, ytr = make_windows((train-mu)/sd, L, H)
dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)  # shuffle windows,
#                                                                       not time itself
class Forecaster(nn.Module):
    def __init__(self, hid=64):
        super().__init__()
        self.rnn  = nn.GRU(1, hid, batch_first=True)   # or nn.LSTM
        self.head = nn.Linear(hid, H)
    def forward(self, x):
        out, _ = self.rnn(x)                  # out: (N, L, hid)
        return self.head(out[:, -1])          # read the LAST step -> (N, H)

model = Forecaster()
opt   = torch.optim.Adam(model.parameters(), lr=1e-3)   # Adam: chapter 8
lossf = nn.MSELoss()
for epoch in range(30):
    for xb, yb in dl:                          # training = teacher forcing:
        opt.zero_grad()                        # the inputs are always TRUE past values
        loss = lossf(model(xb), yb)
        loss.backward(); opt.step()

model.eval()                                   # multi-step forecast = autoregressive rollout
with torch.no_grad():
    ctx = Xtr[-1:].clone()                     # (1, L, 1) last known window
    horizon = 12
    for _ in range(horizon):                   # feed each prediction back in
        yhat = model(ctx)[:, :1]               # errors now COMPOUND step to step
        ctx  = torch.cat([ctx[:, 1:], yhat.view(1, 1, 1)], dim=1)
