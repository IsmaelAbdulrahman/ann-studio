# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 34: Case study: anomaly detection with autoencoders
# Section: In code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-anomaly"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
D, d = 8, 2                                          # 8 sensors, true 2-D structure
A = rng.randn(d, D)                                  # fixed mixing: data near a 2-D plane
def normal(n): return rng.randn(n, d) @ A + 0.10*rng.randn(n, D)
Xtr = normal(500); mu, sd = Xtr.mean(0), Xtr.std(0)
z = lambda X: (X - mu)/sd                            # standardize with train stats

W1 = rng.randn(D, d)*0.1; W2 = rng.randn(d, D)*0.1   # encoder D->2, decoder 2->D
S = z(Xtr)
for ep in range(4000):                               # train AE on NORMAL data only (GD)
    H = S@W1; G = 2*(H@W2 - S)/len(S)                # linear autoencoder (= PCA)
    W2 -= 0.05*(H.T@G); W1 -= 0.05*(S.T@(G@W2.T))
err = lambda X: np.mean((z(X)@W1@W2 - z(X))**2, axis=1)   # reconstruction error / row

thr = np.percentile(err(normal(300)), 99)            # threshold from NORMAL errors only
Xn = normal(200)                                     # held-out normal traffic
Xa = normal(40) + 3.0*rng.randn(40, D)               # anomalies: pushed off the plane
X = np.vstack([Xn, Xa]); lab = np.r_[np.zeros(200), np.ones(40)]
flag = (err(X) > thr).astype(int)
tp = int(((flag==1)&(lab==1)).sum()); fp = int(((flag==1)&(lab==0)).sum())
fn = int(((flag==0)&(lab==1)).sum())
print(f"normal  mean recon-err {err(Xn).mean():.4f}")
print(f"anomaly mean recon-err {err(Xa).mean():.4f}   ({err(Xa).mean()/err(Xn).mean():.0f}x higher)")
print(f"threshold (99th pct of normal) {thr:.4f}")
print(f"TP {tp}  FP {fp}  FN {fn}   ->  precision {tp/(tp+fp):.2f}   recall {tp/(tp+fn):.2f}")
