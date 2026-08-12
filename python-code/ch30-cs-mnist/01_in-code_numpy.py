# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 30: Case study: handwritten-digit recognition (MNIST)
# Section: In code
# Code example 1 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-mnist"
# ====================================================================

import numpy as np
rng = np.random.RandomState(1)

# 8x8 template for the digit 0 (a ring), 64 pixel values in row-major order
T0 = np.array([
 0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0,
 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,1,0,0,0,0,1,0, 0,0,1,1,1,1,0,0],
 dtype=float)

def show(v):                                   # print an 8x8 vector as ASCII art
    for r in range(8):
        print("".join("#" if v[r*8+c] > 0.5 else "." for c in range(8)))

print("clean template for digit 0:")
show(T0)
noisy = np.clip(T0 + rng.normal(0, 0.35, 64), 0, 1)
print("\nthe same digit seen through a noisy sensor:")
show(noisy)
