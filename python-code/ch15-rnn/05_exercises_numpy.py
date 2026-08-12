# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: Exercises
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import numpy as np
def run_rnn(x, Wx=1.0, Wh=1.0, b=0.0, g=lambda z: z, h0=0.0):
    h, hs = h0, []
    for xt in x:
        h = g(Wx*xt + Wh*h + b)
        hs.append(h)
    return np.array(hs)

x = np.array([1., 0., 1., 1.])
print(np.round(run_rnn(x, Wx=2.0, Wh=0.5, b=-0.5, g=np.tanh), 4))
