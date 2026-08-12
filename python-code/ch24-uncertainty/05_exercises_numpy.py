# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 24: Uncertainty, calibration & Bayesian nets
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "uncertainty"
# ====================================================================

import numpy as np
def decompose(P):                        # P: K x C member probabilities
    ent = lambda p: -np.sum(p * np.log(p + 1e-12), axis=-1)
    pbar = P.mean(0)
    total = ent(pbar)                    # entropy of the average
    aleatoric = ent(P).mean()            # average of the entropies
    return total, aleatoric, total - aleatoric

for name, P in [("agree   ", np.array([[0.9, 0.1], [0.9, 0.1]])),
                ("disagree", np.array([[0.99, 0.01], [0.01, 0.99]]))]:
    t, a, e = decompose(P)
    print(f"{name}  total={t:.3f}  aleatoric={a:.3f}  epistemic={e:.3f}")
