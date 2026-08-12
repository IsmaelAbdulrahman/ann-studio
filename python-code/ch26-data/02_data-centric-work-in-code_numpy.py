# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 26: Data-centric deep learning
# Section: Data-centric work in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "data"
# ====================================================================

import numpy as np
np.random.seed(1)

# --- mixup: convex-combine two examples AND their one-hot labels ---
lam = np.random.beta(0.2, 0.2)                 # mixing coefficient ~ Beta(a, a)
xa, xb = np.array([0., 1., 2., 3.]), np.array([4., 3., 2., 1.])
ya, yb = np.array([1., 0.]), np.array([0., 1.])    # one-hot: class 0 vs class 1
xm = lam * xa + (1 - lam) * xb
ym = lam * ya + (1 - lam) * yb
print("lambda   =", round(float(lam), 4))
print("mixed x  =", xm.round(4))
print("mixed y  =", ym.round(4), " (soft label sums to", round(float(ym.sum()), 3), ")")

# --- leakage: fit the scaler on ALL data (wrong) vs on the training split only ---
feat = np.random.normal(10.0, 3.0, 200)        # one feature, full dataset
train, test = feat[:150], feat[150:]
mu_all = feat.mean()                           # WRONG: this mean saw the test rows
mu_tr  = train.mean()                          # RIGHT: training statistics only
print("mean(all) =", round(float(mu_all), 4),
      "  mean(train) =", round(float(mu_tr), 4),
      "  leaked shift =", round(float(mu_all - mu_tr), 4))
print("centered-train mean, GLOBAL stats =", round(float((train - mu_all).mean()), 4))  # != 0
print("centered-train mean, TRAIN  stats =", round(float((train - mu_tr).mean()), 4))   # == 0
