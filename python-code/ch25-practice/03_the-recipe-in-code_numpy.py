# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: The recipe in code
# Code example 3 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np

# fraud detector on 1000 transactions (fraud = the positive class)
TP, FP, FN, TN = 6, 20, 4, 970
N = TP + FP + FN + TN

accuracy  = (TP + TN) / N
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f1        = 2 * precision * recall / (precision + recall)

print(f"accuracy : {accuracy:.4f}")
print(f"precision: {precision:.4f}")
print(f"recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
# a lazy 'always legit' model beats it on accuracy while catching zero fraud
print(f"baseline (predict all legit): {(TN + FP) / N:.4f}")
