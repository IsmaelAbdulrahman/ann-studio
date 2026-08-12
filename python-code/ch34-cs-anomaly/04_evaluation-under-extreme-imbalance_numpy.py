# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 34: Case study: anomaly detection with autoencoders
# Section: Evaluation under extreme imbalance
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-anomaly"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
D, d = 8, 2                                          # 8 sensors, intrinsic dim 2
A = rng.randn(d, D)
def normal(n): return rng.randn(n, d) @ A + 0.10*rng.randn(n, D)
Xtr = normal(500); mu, sd = Xtr.mean(0), Xtr.std(0)
z = lambda X: (X - mu)/sd                            # standardize with train stats

# train the linear autoencoder in CLOSED FORM: it is just PCA (SVD, no gradient descent)
U, S, Vt = np.linalg.svd(z(Xtr), full_matrices=False)
P = Vt[:d]                                           # top-d principal axes (d x D)
err = lambda X: np.mean((z(X)@P.T@P - z(X))**2, axis=1)   # residual off the subspace

thr = np.percentile(err(normal(300)), 99)            # threshold from NORMAL errors only
Xn = normal(990)                                     # a realistic 2%-anomaly stream:
Xa = normal(20) + 0.30*rng.randn(20, D)              # 20 SUBTLE anomalies, barely off-plane
X  = np.vstack([Xn, Xa]); lab = np.r_[np.zeros(990), np.ones(20)]
e  = err(X)

flag = (e > thr).astype(int)
tp = int(((flag==1)&(lab==1)).sum()); fp = int(((flag==1)&(lab==0)).sum()); fn = int(((flag==0)&(lab==1)).sum())
prec = tp/(tp+fp); rec = tp/(tp+fn)
k = 20; patk = lab[np.argsort(-e)[:k]].mean()        # precision@k: an analyst's daily budget
order = np.argsort(-e); ls = lab[order]              # PR-AUC (average precision) by ranking
tpc = np.cumsum(ls); fpc = np.cumsum(1-ls)
Pc = tpc/(tpc+fpc); Rc = tpc/lab.sum()
ap = np.sum((Rc - np.r_[0.0, Rc[:-1]]) * Pc)

print(f"anomaly base rate            {lab.mean()*100:.1f}%")
print(f"accuracy  detector {(flag==lab).mean():.3f}   vs 'flag nothing' {1-lab.mean():.3f}")
print(f"precision {prec:.2f}    recall {rec:.2f}    (TP {tp}  FP {fp}  FN {fn})")
print(f"precision@{k}  {patk:.2f}     PR-AUC {ap:.3f}")
