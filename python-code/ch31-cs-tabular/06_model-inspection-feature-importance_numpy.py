# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: Model inspection &amp; feature importance
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n = 1000
tenure=rng.uniform(0,60,n); charges=rng.uniform(20,120,n); calls=rng.poisson(1.5,n)
z=-1.8-0.07*tenure+0.02*charges+0.55*calls+rng.normal(0,0.5,n)
y=(rng.rand(n)<1/(1+np.exp(-z))).astype(float)
X=np.stack([tenure,charges,calls],1)
Xtr,ytr,Xte,yte=X[:800],y[:800],X[800:],y[800:]
mu,sd=Xtr.mean(0),Xtr.std(0); Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
w=np.zeros(3); b=0.0
for _ in range(400):
    p=1/(1+np.exp(-(Xtr@w+b))); g=p-ytr
    w-=0.3*(Xtr.T@g)/len(ytr); b-=0.3*g.mean()

def auc(sc, lab):                                    # AUC by the ranking rule
    pos,neg=sc[lab==1],sc[lab==0]
    return ((pos[:,None]>neg[None,:]).sum()+0.5*(pos[:,None]==neg[None,:]).sum())/(pos.size*neg.size)

base=auc(1/(1+np.exp(-(Xte@w+b))), yte)
print(f"test ROC-AUC (all features) = {base:.3f}")
names=['tenure','charges','calls']; perm=np.random.RandomState(42)
for j in range(3):
    Xp=Xte.copy(); Xp[:,j]=perm.permutation(Xp[:,j])  # break feature j's link to y
    drop=base-auc(1/(1+np.exp(-(Xp@w+b))), yte)
    print(f"  shuffle {names[j]:8s}: AUC drop = {drop:+.3f}")
