# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: In code
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n = 1000
tenure  = rng.uniform(0, 60, n)          # months as a customer
charges = rng.uniform(20, 120, n)        # monthly bill, $
calls   = rng.poisson(1.5, n)            # support calls last quarter

# ground-truth log-odds of churn (short tenure, high bill, many calls -> churn)
z = -1.8 - 0.07*tenure + 0.02*charges + 0.55*calls + rng.normal(0, 0.5, n)
y = (rng.rand(n) < 1/(1+np.exp(-z))).astype(float)
X = np.stack([tenure, charges, calls], axis=1)

print(f"customers: {n}   churned: {int(y.sum())}  ({y.mean()*100:.1f}%)")
print(f"majority-class baseline accuracy: {max(y.mean(), 1-y.mean()):.3f}\n")
print(f"{'group':8s}{'tenure':>9s}{'charges':>9s}{'calls':>8s}")
for lab, name in [(0, 'stayed'), (1, 'churned')]:
    m = X[y == lab].mean(0)
    print(f"{name:8s}{m[0]:9.1f}{m[1]:9.1f}{m[2]:8.2f}")
