# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: In code
# Code example 3 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n = 1000
tenure  = rng.uniform(0, 60, n)
charges = rng.uniform(20, 120, n)
calls   = rng.poisson(1.5, n)
z = -1.8 - 0.07*tenure + 0.02*charges + 0.55*calls + rng.normal(0, 0.5, n)
y = (rng.rand(n) < 1/(1+np.exp(-z))).astype(float)
X = np.stack([tenure, charges, calls], axis=1)

# split, then STANDARDIZE using training statistics only (no leakage)
Xtr, ytr, Xte, yte = X[:800], y[:800], X[800:], y[800:]
mu, sd = Xtr.mean(0), Xtr.std(0)
Xtr = (Xtr - mu) / sd
Xte = (Xte - mu) / sd

# logistic regression from scratch (one sigmoid neuron, BCE loss)
w = np.zeros(3); b = 0.0
for ep in range(1, 401):
    p  = 1 / (1 + np.exp(-(Xtr @ w + b)))
    g  = p - ytr                              # gradient of BCE wrt the logit
    w -= 0.3 * (Xtr.T @ g) / len(ytr)
    b -= 0.3 * g.mean()
    if ep in (1, 50, 100, 200, 400):
        print(f"epoch {ep:3d}   train acc {((p>0.5)==ytr).mean():.3f}")

pt   = 1 / (1 + np.exp(-(Xte @ w + b)))
pred = pt > 0.5
tp = np.sum(pred & (yte==1)); fp = np.sum(pred & (yte==0)); fn = np.sum(~pred & (yte==1))
print(f"\ntest accuracy {(pred==yte).mean():.3f}   (majority baseline {max(yte.mean(),1-yte.mean()):.3f})")
print(f"precision {tp/(tp+fp):.3f}   recall {tp/(tp+fn):.3f}")
print("\nstandardized weights (effect of a 1-sd change):")
for name, wi in sorted(zip(['tenure','charges','calls'], w), key=lambda t: -abs(t[1])):
    print(f"  {name:8s} {wi:+.3f}")
