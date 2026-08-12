# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 28: Interpretability & explainability
# Section: Attribution in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "interpretability"
# ====================================================================

import numpy as np
np.random.seed(0)                              # determinism
n = 3
w = np.random.randn(n)                         # linear weights
A = np.random.randn(n, n); A = (A + A.T) / 2   # symmetric interaction matrix
x = np.random.randn(n)                         # the input we want to explain
b = np.zeros(n)                                # baseline / reference input

def perms(seq):                                # all orderings, using no imports
    if len(seq) <= 1: return [list(seq)]
    out = []
    for i in range(len(seq)):
        for p in perms(seq[:i] + seq[i+1:]):
            out.append([seq[i]] + p)
    return out

def shapley(f):                                # exact Shapley: average over orderings
    orders = perms(list(range(n)))
    phi = np.zeros(n)
    for order in orders:
        z = b.copy(); prev = f(z)              # start from the baseline coalition
        for i in order:
            z = z.copy(); z[i] = x[i]          # feature i joins the coalition
            cur = f(z); phi[i] += cur - prev   # its marginal contribution
            prev = cur
    return phi / len(orders)

lin   = lambda z: w @ z                        # game 1: the linear part
inter = lambda z: z @ A @ z                    # game 2: the interaction part
full  = lambda z: lin(z) + inter(z)            # the whole model

phi = shapley(full)
print("Shapley values     =", np.round(phi, 4))
print("sum of Shapley      =", round(float(phi.sum()), 4))
print("f(x) - f(baseline)  =", round(float(full(x) - full(b)), 4))  # efficiency: equal
print("additivity holds    =", np.allclose(shapley(lin) + shapley(inter), phi))
