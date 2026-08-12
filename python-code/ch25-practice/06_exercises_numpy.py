# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 25: Training in practice: the complete recipe
# Section: Exercises
# Code example 6 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "practice"
# ====================================================================

import numpy as np
# confusion matrix rows = actual, cols = predicted:  [[TN, FP], [FN, TP]]
cm = np.array([[45, 5],
               [8, 42]])
TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
prec = TP / (TP + FP)
rec  = TP / (TP + FN)
f1   = 2 * prec * rec / (prec + rec)
print(f"precision {prec:.4f}   recall {rec:.4f}   F1 {f1:.4f}")
