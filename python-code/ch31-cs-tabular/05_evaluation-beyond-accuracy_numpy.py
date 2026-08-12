# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: Evaluation beyond accuracy
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import numpy as np
rng = np.random.RandomState(7)
n = 60
y = (rng.rand(n) < 0.35).astype(int)                       # true churn labels
# an imperfect model's scores in [0,1]: churners bumped up, but heavy overlap
scores = np.clip(0.55*rng.rand(n) + 0.20*y + 0.20*rng.randn(n) + 0.20, 0.02, 0.98)

def prf(y, pred):
    tp = np.sum((pred==1)&(y==1)); fp = np.sum((pred==1)&(y==0)); fn = np.sum((pred==0)&(y==1))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    return prec, rec, f1

print(f"churners: {int(y.sum())} of {n}  ({y.mean()*100:.0f}%)")
print("thr    prec   rec    F1")
best_thr, best_f1 = 0.0, -1.0
for thr in np.arange(0.30, 0.76, 0.05):
    pr, rc, f1 = prf(y, (scores>=thr).astype(int))
    if f1 > best_f1: best_thr, best_f1 = thr, f1
    print(f"{thr:.2f}   {pr:.3f}  {rc:.3f}  {f1:.3f}")
print(f"best-F1 threshold = {best_thr:.2f}   (F1 = {best_f1:.3f})")

# ROC-AUC = P(score(random churner) > score(random stayer)); ties count 1/2
pos, neg = scores[y==1], scores[y==0]
wins = (pos[:,None] > neg[None,:]).sum() + 0.5*(pos[:,None] == neg[None,:]).sum()
print(f"ROC-AUC = {wins/(pos.size*neg.size):.3f}   (0.5 coin flip, 1.0 perfect)")
