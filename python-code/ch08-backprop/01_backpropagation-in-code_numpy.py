# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 8: Backpropagation
# Section: Backpropagation in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "backprop"
# ====================================================================

import numpy as np
np.set_printoptions(precision=4, suppress=True)
sig = lambda z: 1/(1+np.exp(-z))

x  = np.array([0.05, 0.10])                 # input  (row vector a^(0))
y  = 0.70                                    # target
W1 = np.array([[0.15, 0.25],                 # 2 inputs -> 2 hidden units
               [0.20, 0.30]])
b1 = np.array([0.35, 0.35])
W2 = np.array([[0.40], [0.50]])              # 2 hidden -> 1 linear output
b2 = np.array([0.60])

def forward(W1, b1, W2, b2):                  # returns z1, a1, yhat, L
    z1 = x @ W1 + b1;  a1 = sig(z1)
    yhat = a1 @ W2 + b2                        # linear output, g' = 1
    return z1, a1, yhat, 0.5 * (yhat - y) ** 2

z1, a1, yhat, L = forward(W1, b1, W2, b2)
print("z1   =", z1, "  a1 =", a1)
print("yhat =", yhat, "  L =", L)

d2  = yhat - y                                # delta^(2)  (linear g'=1)
gW2 = np.outer(a1, d2)                         # dL/dW2
d1  = (d2 @ W2.T) * a1 * (1 - a1)             # delta^(1)  (sigmoid g')
gW1 = np.outer(x, d1)                          # dL/dW1
print("dL/dW2 =", gW2.ravel(), "  dL/db2 =", d2)
print("dL/dW1 =", gW1.ravel(), "  dL/db1 =", d1)

# finite-difference check on the output weight W2[0]
e = 1e-5;  Wp = W2.copy(); Wp[0] += e;  Wm = W2.copy(); Wm[0] -= e
num = (forward(W1, b1, Wp, b2)[3] - forward(W1, b1, Wm, b2)[3]) / (2 * e)
print(f"gW2[0]: analytic {gW2[0,0]:.6f}   numeric {num[0]:.6f}")
