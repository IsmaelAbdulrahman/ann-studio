# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 5: Features, embeddings & what a network learns
# Section: Representations in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "representations"
# ====================================================================

import numpy as np
np.random.seed(0)

# concentric rings: inner ring = class -1, outer ring = class +1
def ring(n, r):
    t = np.random.uniform(0, 2*np.pi, n)
    rad = r + 0.12*np.random.randn(n)
    return np.c_[rad*np.cos(t), rad*np.sin(t)]

X = np.vstack([ring(200, 1.0), ring(200, 3.0)])
y = np.r_[-np.ones(200), np.ones(200)]            # labels -1 / +1

def lin_acc(F):                                    # least-squares linear classifier
    Fb = np.c_[F, np.ones(len(F))]                 # append a bias feature
    w, *_ = np.linalg.lstsq(Fb, y, rcond=None)
    return (np.sign(Fb @ w) == y).mean()

raw    = lin_acc(X)                                # raw (x, y): not separable
phi    = np.c_[X, (X**2).sum(1)]                   # add r^2 = x^2 + y^2
lifted = lin_acc(phi)                              # now linearly separable
print("linear accuracy, raw (x,y)   =", round(float(raw), 3))      # ~0.52 (chance)
print("linear accuracy, lifted +r^2 =", round(float(lifted), 3))   # 1.0

# the lift makes the classes occupy disjoint r^2 intervals -> a threshold separates
r2 = (X**2).sum(1)
print("max r^2 inner =", round(float(r2[y < 0].max()), 2),
      "<  min r^2 outer =", round(float(r2[y > 0].min()), 2))
