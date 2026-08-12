# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Exercises
# Code example 22 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np
def softmax(z):
    z = z - z.max(axis=1, keepdims=True); e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
logits = np.array([[2.0, 0.5, 0.1],
                   [0.2, 1.5, 0.3],
                   [0.1, 0.2, 2.0]])
labels = np.array([0, 1, 2])
P = softmax(logits)
picked = P[np.arange(len(labels)), labels]   # fancy indexing: one prob per row
loss = -np.log(picked).mean()
print("correct-class probs:", picked.round(3))   # [0.728 0.635 0.761]
print("cross-entropy      :", round(float(loss), 4))   # 0.348
