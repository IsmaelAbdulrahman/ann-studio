# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 22: Transfer learning & fine-tuning
# Section: Transfer learning in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "transfer"
# ====================================================================

import numpy as np
np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

d, r, alpha = 1024, 8, 16          # weight is d x d; LoRA rank r; scale alpha
x  = np.random.randn(d)            # an input activation vector
W  = 0.02 * np.random.randn(d, d)  # frozen pretrained weight (never updated)
A  = 0.01 * np.random.randn(r, d)  # down-projection  (r x d)
B  = np.zeros((d, r))              # up-projection init to ZERO  (d x r)

# LoRA forward: (W + (alpha/r) B@A) @ x   vs   the base   W @ x
y_base = W @ x
y_lora = (W + (alpha / r) * (B @ A)) @ x
print("at init  max|y_lora - y_base| =", np.max(np.abs(y_lora - y_base)))  # exactly 0.0

# after a few training steps B is small but non-zero:
B  = 0.01 * np.random.randn(d, r)
dW = (alpha / r) * (B @ A)                       # the effective weight update
print("rank(B@A) =", np.linalg.matrix_rank(B @ A), " (<= r =", r, ")")   # 8
print("||dW|| / ||W|| =", round(np.linalg.norm(dW) / np.linalg.norm(W), 4))  # small: 0.0281

full = d * d
lora = 2 * d * r
print("full params  =", full)                    # 1048576
print("LoRA params  =", lora, "=", round(100 * lora / full, 2), "% of full")  # 16384 = 1.56 %
