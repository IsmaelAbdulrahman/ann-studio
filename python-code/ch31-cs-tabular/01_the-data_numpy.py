# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 31: Case study: tabular classification & churn
# Section: The data
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-tabular"
# ====================================================================

import numpy as np
# a categorical column: contract type as integer codes 0/1/2
contract = np.array([0, 2, 1, 0, 1, 2])
onehot = np.eye(3)[contract]                 # 3 columns, exactly one 1 per row
print("one-hot 'contract':\n", onehot.astype(int))
# a numeric column: standardize to mean 0, sd 1 (fit on TRAINING data only)
charges = np.array([20., 55., 95., 70., 120., 45.])
z = (charges - charges.mean()) / charges.std()
print("standardized charges:", np.round(z, 2))
print("check -> mean", round(z.mean(), 3), " sd", round(z.std(), 3))
