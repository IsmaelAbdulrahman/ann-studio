# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: The training loop
# Code example 9 of 15 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import numpy as np

# minimize L(w) = (w - 3)**2 ;  dL/dw = 2*(w - 3)
w, lr = 0.0, 0.1
for step in range(1, 21):
    grad = 2 * (w - 3)          # "backward"
    w = w - lr * grad           # "step"
    if step % 5 == 0:
        print(f"step {step:2d}: w = {w:.4f}  loss = {(w - 3)**2:.5f}")
print("converged near w =", round(w, 4), "(target 3)")
