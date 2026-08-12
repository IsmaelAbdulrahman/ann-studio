# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 2: The perceptron & linear separability
# Section: Exercises
# Code example 6 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "perceptron"
# ====================================================================

import numpy as np
rng = np.random.RandomState(19)
# Two heavily overlapping clouds -> NOT linearly separable.
X = np.vstack([rng.randn(30, 2) + [0.8, 0.8],
               rng.randn(30, 2) + [-0.8, -0.8]])
y = np.array([1]*30 + [0]*30)
def acc(w, b):
    return np.mean([(1 if w @ xi + b >= 0 else 0) == yi for xi, yi in zip(X, y)])
w = np.zeros(2); b = 0.0
best_w, best_b, best_run, run = w.copy(), b, -1, 0
for epoch in range(20):
    for xi, yi in zip(X, y):
        e = yi - (1 if w @ xi + b >= 0 else 0)
        if e:                                   # a mistake -> keep learning
            w = w + e * xi; b = b + e; run = 0
        else:                                   # a correct streak
            run += 1
            if run > best_run:                  # longest streak -> pocket it
                best_run, best_w, best_b = run, w.copy(), b
print("final-weights accuracy:", round(acc(w, b), 3))       # 0.617
print("pocket   accuracy:", round(acc(best_w, best_b), 3))  # 0.967
    
