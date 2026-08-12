# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: The RNN in code
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import numpy as np
# BPTT gradient magnitude across T steps ~ (||Wh|| * |g'|)^T
T, gp = 50, 0.9                         # gp = a representative tanh slope |g'|
for Wh in [0.8, 1.0, 1.2]:
    base = Wh * gp
    print(f"||Wh||={Wh}  base={base:.2f}  (base)^{T} = {base**T:.2e}")
