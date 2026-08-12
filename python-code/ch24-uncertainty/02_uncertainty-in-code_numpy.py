# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 24: Uncertainty, calibration & Bayesian nets
# Section: Uncertainty in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "uncertainty"
# ====================================================================

import numpy as np
np.random.seed(0)

def softmax_rows(Z):
    Z = Z - Z.max(axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=-1, keepdims=True)

def predictive(base, noise, K=500):           # K ensemble members = noisy logit copies
    Z = base + noise * np.random.randn(K, base.size)
    P = softmax_rows(Z)                        # K x C member probabilities
    pbar = P.mean(0)                           # predictive mean
    H = -np.sum(pbar * np.log(pbar + 1e-12))   # predictive entropy (nats)
    return pbar, P.var(0).sum(), H             # mean, total variance, entropy

pbar_id,  var_id,  H_id  = predictive(np.array([3.0, 1.0, 0.2]), 0.4)  # in-distribution
pbar_ood, var_ood, H_ood = predictive(np.array([0.3, 0.1, 0.0]), 2.5)  # OOD-like input

print("ID : mean", np.round(pbar_id, 3),  " var", round(var_id, 4),  " entropy", round(H_id, 3))
print("OOD: mean", np.round(pbar_ood, 3), " var", round(var_ood, 4), " entropy", round(H_ood, 3))
print("epistemic spread larger for OOD:", bool(var_ood > var_id),
      "  entropy higher:", bool(H_ood > H_id))
