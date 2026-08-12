# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: The RNN in code
# Code example 4 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import numpy as np
sig = lambda z: 1/(1+np.exp(-z))
x, h, c = 1.0, 0.5, 0.8                       # x_t, h_{t-1}, c_{t-1}
f  = sig(x*1.0 + h*1.0 + 0.5)                 # forget gate   (Wx, Wh, b)
i  = sig(x*0.5 + h*0.5 - 1.0)                 # input  gate
o  = sig(x*1.0 + h*0.0 + 0.0)                 # output gate
ct = np.tanh(x*1.0 + h*1.0 + 0.0)            # candidate  c-tilde
c_new = f*c + i*ct                            # additive cell update:  f*c + i*c~
h_new = o*np.tanh(c_new)
print("f,i,o,cand =", np.round([f, i, o, ct], 4))          # [0.8808 0.4378 0.7311 0.9051]
print("c_t =", round(c_new, 4), " h_t =", round(h_new, 4))  # 1.1009  0.5855
