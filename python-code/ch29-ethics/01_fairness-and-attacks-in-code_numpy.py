# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 29: Ethics, fairness & safety
# Section: Fairness and attacks in code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "ethics"
# ====================================================================

import numpy as np

# Per-group confusion counts from one calibrated risk model at a common threshold.
# Spell out every person as (label y, prediction yhat, group g).
def people(tp, fn, fp, tn, g):
    y    = np.r_[np.ones(tp+fn), np.zeros(fp+tn)]                        # true labels
    yhat = np.r_[np.ones(tp), np.zeros(fn), np.ones(fp), np.zeros(tn)]   # predictions
    return y, yhat, np.full(tp+fn+fp+tn, g)

yA, hA, gA = people(600, 400, 200,  800, 0)   # group A: base rate 1000/2000 = 0.50
yB, hB, gB = people(150, 350,  50, 1450, 1)   # group B: base rate  500/2000 = 0.25
y, yhat, g = np.r_[yA, yB], np.r_[hA, hB], np.r_[gA, gB]

def rates(y, yhat):
    pos, neg = (y == 1), (y == 0)
    sel = yhat.mean()                 # selection rate  P(Yhat=1)      -> demographic parity
    tpr = yhat[pos].mean()            # true-positive   P(Yhat=1|Y=1)  -> equal opportunity
    fpr = yhat[neg].mean()            # false-positive  P(Yhat=1|Y=0)  -> equalized odds
    ppv = y[yhat == 1].mean()         # precision       P(Y=1|Yhat=1)  -> predictive parity
    return np.array([sel, tpr, fpr, ppv])

A, B = rates(y[g==0], yhat[g==0]), rates(y[g==1], yhat[g==1])
print("group A: sel=%.3f TPR=%.3f FPR=%.3f PPV=%.3f" % tuple(A))
print("group B: sel=%.3f TPR=%.3f FPR=%.3f PPV=%.3f" % tuple(B))
print("demographic-parity diff |selA-selB| =", round(abs(A[0]-B[0]), 3))   # 0.300
print("equal-opportunity  gap  |TPRA-TPRB| =", round(abs(A[1]-B[1]), 3))   # 0.300
print("equalized-odds FPR gap  |FPRA-FPRB| =", round(abs(A[2]-B[2]), 3))   # 0.167
print("predictive-parity  gap  |PPVA-PPVB| =", round(abs(A[3]-B[3]), 3))   # 0.000
