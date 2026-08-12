# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import numpy as np
sig = lambda z: 1/(1+np.exp(-z))
x, h = 1.0, 0.5
z = sig(x*0.5 + h*0.5 - 0.5)                   # update gate
r = sig(x*1.0 + h*1.0 + 0.0)                   # reset  gate
hc = np.tanh(x*1.0 + (r*h)*1.0 + 0.0)         # candidate uses r ⊙ h
h_new = (1 - z)*h + z*hc                        # interpolate old state and candidate
print("z,r,cand =", np.round([z, r, hc], 4))   # [0.5622 0.8176 0.8872]
print("h_t =", round(h_new, 4))                # 0.7177
