# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 28: Interpretability & explainability
# Section: Attribution in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "interpretability"
# ====================================================================

import numpy as np
np.random.seed(0)                                  # determinism
def f(x):                                          # a small differentiable toy model
    return 2*x[0] + 0*x[1] - x[2] + x[3]**2        # two linear feats, one squared feat
x    = np.array([1.0, 5.0, 2.0, 3.0])              # the input we want to explain
base = np.zeros(4)                                 # reference used for occlusion

eps = 1e-5                                          # (1) vanilla-gradient saliency
grad = np.zeros(4)                                 #     by central finite differences
for i in range(4):
    xp = x.copy(); xp[i] += eps
    xm = x.copy(); xm[i] -= eps
    grad[i] = (f(xp) - f(xm)) / (2*eps)            # d f / d x_i  at the input
print("f(x)              =", f(x))                 # 9.0
print("gradient saliency =", np.round(grad, 4))    # [ 2  0 -1  6 ]  local slope
print("|gradient|        =", np.round(np.abs(grad), 4))

occ = np.zeros(4)                                  # (2) occlusion-importance map
for i in range(4):
    xo = x.copy(); xo[i] = base[i]                 # reset feature i to the baseline
    occ[i] = f(x) - f(xo)                          # how much the output drops
print("occlusion map     =", np.round(occ, 4))     # [ 2  0 -2  9 ]  remove-feature
