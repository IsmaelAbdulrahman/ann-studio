# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Exercises
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import numpy as np
def softmax(v):
    v = v - v.max(); e = np.exp(v); return e / e.sum()
q = np.array([1.0, 0.0])
K = np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.]])   # added k4 = [0,0]
V = np.array([[1.,0.],[0.,1.],[1.,1.],[2.,2.]])   # added v4 = [2,2]
a = softmax((K @ q) / np.sqrt(2))
print("weights:", a.round(4))
print("output :", (a @ V).round(4))
