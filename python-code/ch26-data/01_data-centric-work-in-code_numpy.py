# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 26: Data-centric deep learning
# Section: Data-centric work in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "data"
# ====================================================================

import numpy as np
np.random.seed(0)

# --- inverse-frequency class weights, normalized so the per-example average is 1 ---
counts = np.array([900, 100])          # [negatives, positives]
N, K = counts.sum(), len(counts)
w = N / (K * counts)                   # w_c = N / (K * n_c)
print("class weights         =", w.round(4))            # [0.5556 5.    ]
print("ratio w[pos]/w[neg]   =", round(float(w[1]/w[0]), 3))   # 9.0 == 900/100

avg = (counts * w).sum() / N           # average weight per example
print("avg weight per example =", round(float(avg), 4))  # 1.0  (that is the normalization)

mass = counts * w                      # total weight each class puts on the loss/gradient
print("weighted mass per class =", mass.round(1))        # [500. 500.] -> rebalanced

# --- focal loss vs cross-entropy across a range of p_t (gamma = 2) ---
gamma = 2.0
pt = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
ce = -np.log(pt)                       # cross-entropy of the true class
fl = (1 - pt)**gamma * ce              # focal loss FL = -(1-pt)^gamma * log pt
print("p_t          :", pt)
print("cross-entropy:", ce.round(4))
print("focal (g=2)  :", fl.round(4))
print("modulator    :", ((1 - pt)**gamma).round(4))      # 0.01 at pt=0.9, 0.49 at pt=0.3
