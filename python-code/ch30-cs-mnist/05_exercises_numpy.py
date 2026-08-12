# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 30: Case study: handwritten-digit recognition (MNIST)
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-mnist"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
T = np.array([
 [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0,
  0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0],
 [0,0,0,1,1,0,0,0, 0,0,1,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0,
  0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,0,1,1,0,0,0, 0,0,1,1,1,1,0,0],
 [1,1,1,1,1,1,0,0, 0,0,0,0,0,1,0,0, 0,0,0,0,1,0,0,0, 0,0,0,1,0,0,0,0,
  0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0]], dtype=float)
def make(n):
    X,y=[],[]
    for k in range(3):
        b=np.tile(T[k],(n,1)); f=(rng.rand(n,64)<0.10).astype(float)
        img=np.clip(b*(1-f)+(1-b)*f+rng.normal(0,0.6,(n,64)),0,1)
        X.append(img); y.append(np.full(n,k))
    X,y=np.vstack(X),np.concatenate(y); p=rng.permutation(len(y)); return X[p],y[p]
Xtr,ytr=make(60); Xte,yte=make(30); Y=np.eye(3)[ytr]
W1=rng.randn(64,24)*.1;b1=np.zeros(24);W2=rng.randn(24,12)*.1;b2=np.zeros(12);W3=rng.randn(12,3)*.1;b3=np.zeros(3)
for ep in range(120):
    H1=np.maximum(0,Xtr@W1+b1); H2=np.maximum(0,H1@W2+b2)
    S=H2@W3+b3; S-=S.max(1,keepdims=True); P=np.exp(S); P/=P.sum(1,keepdims=True)
    dS=(P-Y)/len(Xtr); dW3=H2.T@dS; db3=dS.sum(0)
    dH2=(dS@W3.T)*(H2>0); dW2=H1.T@dH2; db2=dH2.sum(0)
    dH1=(dH2@W2.T)*(H1>0); dW1=Xtr.T@dH1; db1=dH1.sum(0)
    for par,g in [(W1,dW1),(b1,db1),(W2,dW2),(b2,db2),(W3,dW3),(b3,db3)]: par-=0.25*g
pred=lambda X: (np.maximum(0,np.maximum(0,X@W1+b1)@W2+b2)@W3+b3).argmax(1)
print("test accuracy:", round((pred(Xte)==yte).mean(),3))
