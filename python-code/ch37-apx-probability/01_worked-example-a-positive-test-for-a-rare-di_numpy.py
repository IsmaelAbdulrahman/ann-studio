# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: Conditional probability and Bayes' theorem → Worked example — a positive test for a rare disease
# Code example 1 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
prev, sens, spec = 0.01, 0.99, 0.95        # prevalence, sensitivity, specificity
N = 200000
disease  = rng.binomial(1, prev, N).astype(bool)         # who is truly ill
pos_ill  = rng.binomial(1, sens, N).astype(bool)         # test + given ill
pos_well = rng.binomial(1, 1 - spec, N).astype(bool)     # test + given healthy
positive = np.where(disease, pos_ill, pos_well)
emp = disease[positive].mean()             # empirical P(disease | positive)
den = sens * prev + (1 - spec) * (1 - prev)   # law of total probability -> P(+)
print("evidence P(+)      =", round(den, 4))            # 0.0594
print("posterior (Bayes)  =", round(sens*prev/den, 4))  # 0.1667
print("posterior (sim)    =", round(float(emp), 4))     # ~0.17
