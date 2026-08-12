# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 33: Case study: time-series forecasting
# Section: In code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-sequence"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
T = 360
t = np.arange(T)
series = np.sin(2*np.pi*t/24) + 0.02*t + 0.30*rng.randn(T)   # season + trend + noise

L = 12                                       # look-back window length
X = np.stack([series[i:i+L] for i in range(T-L)])
y = series[L:]                               # one-step-ahead target
n_tr = 250                                   # chronological split: past -> future
Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

mu, sd = Xtr.mean(0), Xtr.std(0)             # standardize with TRAIN stats only
Xtr, Xte = (Xtr-mu)/sd, (Xte-mu)/sd
ym = ytr.mean()                              # center target; add back at the end

w = np.zeros(L); b = 0.0                      # linear AR, trained from scratch (GD)
for ep in range(3000):
    e = (Xtr@w + b) - (ytr - ym)
    w -= 0.01*(2*Xtr.T@e/len(e)); b -= 0.01*(2*e.mean())
pred = lambda Z: Z@w + b + ym
tr = np.mean((pred(Xtr)-ytr)**2); te = np.mean((pred(Xte)-yte)**2)
print(f"train MSE {tr:.3f}   test MSE {te:.3f}   (noise floor ~ {0.30**2:.3f})")
for i in range(4):
    print(f"  t={n_tr+L+i}:  actual {yte[i]:+.2f}   predicted {pred(Xte)[i]:+.2f}")
