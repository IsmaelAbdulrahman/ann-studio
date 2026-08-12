# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 34: Case study: anomaly detection with autoencoders
# Section: In code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-anomaly"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
D, d = 8, 2; A = rng.randn(d, D)
def normal(n): return rng.randn(n, d) @ A + 0.10*rng.randn(n, D)
Xtr = normal(500); mu, sd = Xtr.mean(0), Xtr.std(0); z = lambda X: (X - mu)/sd
W1 = rng.randn(D, d)*0.1; W2 = rng.randn(d, D)*0.1; S = z(Xtr)
for ep in range(4000):
    H = S@W1; G = 2*(H@W2 - S)/len(S); W2 -= 0.05*(H.T@G); W1 -= 0.05*(S.T@(G@W2.T))
err = lambda X: np.mean((z(X)@W1@W2 - z(X))**2, axis=1)

val = err(normal(300))                               # normal error distribution
Xn = normal(200); Xa = normal(40) + 3.0*rng.randn(40, D)
X = np.vstack([Xn, Xa]); lab = np.r_[np.zeros(200), np.ones(40)]; e = err(X)
print("percentile  threshold  precision  recall")
for pct in (90, 95, 99, 99.9):
    thr = np.percentile(val, pct); fl = e > thr
    tp = int((fl & (lab==1)).sum()); fp = int((fl & (lab==0)).sum()); fn = int((~fl & (lab==1)).sum())
    print(f"   {pct:5}    {thr:.4f}     {tp/(tp+fp):.2f}      {tp/(tp+fn):.2f}")
