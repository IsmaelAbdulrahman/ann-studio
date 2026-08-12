# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Exercises
# Code example 6 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
d = 64
scores = rng.randn(10, d) @ rng.randn(d)     # 10 raw dot-product scores
def softmax(v):
    v = v - v.max(); e = np.exp(v); return e / e.sum()
def entropy(p):
    p = p[p > 1e-12]; return -(p * np.log(p)).sum()
print("raw score std   :", round(scores.std(), 2), "(~ sqrt(64)=8)")
print("unscaled entropy:", round(entropy(softmax(scores)), 3))
print("scaled   entropy:", round(entropy(softmax(scores / np.sqrt(d))), 3))
print("max entropy ln10:", round(np.log(10), 3))
