# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0); n = 1000
tenure=rng.uniform(0,60,n); charges=rng.uniform(20,120,n); calls=rng.poisson(1.5,n)
z=-1.8-0.07*tenure+0.02*charges+0.55*calls+rng.normal(0,0.5,n)
y=(rng.rand(n)<1/(1+np.exp(-z))).astype(float)
X=np.stack([tenure,charges,calls],1); Xtr,ytr=X[:800],y[:800]
mu,sd=Xtr.mean(0),Xtr.std(0); Xtr=(Xtr-mu)/sd
def train(lam):
    w=np.zeros(3); b=0.0
    for _ in range(400):
        p=1/(1+np.exp(-(Xtr@w+b))); g=p-ytr
        w-=0.3*((Xtr.T@g)/len(ytr)+lam*w); b-=0.3*g.mean()
    return w
for lam in [0.0, 0.05, 0.3]:
    w=train(lam)
    print(f"lambda={lam:4}: weights={np.round(w,3)}  sum|w|={abs(w).sum():.3f}")
