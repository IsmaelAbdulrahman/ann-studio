# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: The RNN in code
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import numpy as np
def run_rnn(x, Wx=1.0, Wh=1.0, b=0.0, g=lambda z: z, h0=0.0):
    h, hs = h0, []
    for xt in x:
        h = g(Wx*xt + Wh*h + b)      # h_t = g(x_t Wx + h_{t-1} Wh + b)
        hs.append(h)
    return np.array(hs)

x = np.array([1., 0., 1., 1.])
print("perfect memory (Wh=1, identity):", run_rnn(x, Wh=1.0))
print("fading  memory (Wh=0.5, tanh)  :", np.round(run_rnn(x, Wh=0.5, g=np.tanh), 4))
