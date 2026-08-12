# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 30: Case study: handwritten-digit recognition (MNIST)
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-mnist"
# ====================================================================

import numpy as np
# true test labels and a classifier's predictions over the ten digits.
# Every mistake here is one of the notorious look-alike pairs: 4<->9, 3<->5, 7<->1.
y_true = np.array([0,1,2,3,4,5,6,7,8,9, 1,3,4,5,7,9, 4,9,3,5,7,1])
y_pred = np.array([0,1,2,3,4,5,6,7,8,9, 7,5,9,3,1,4, 4,9,3,5,7,1])
K = 10
C = np.zeros((K, K), int)
for t, p in zip(y_true, y_pred):        # build the confusion matrix
    C[t, p] += 1
acc    = np.trace(C) / C.sum()          # correct predictions are the diagonal
recall = np.diag(C) / np.maximum(C.sum(1), 1)     # row-normalized
print("overall accuracy =", round(float(acc), 3))     # 0.727
print("off-diagonal errors (true -> pred):")
for t in range(K):
    for p in range(K):
        if t != p and C[t, p]:
            print(f"   {t} -> {p}")
for k in (1, 3, 4, 5, 7, 9):
    print(f"recall[{k}] = {recall[k]:.2f}")           # each 0.67
    
