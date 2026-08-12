# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 43: Appendix I · Formulas & symbols quick reference
# Section: Activation functions &amp; their derivatives — Ch.4
# Code example 1 of 1 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-formulas"
# ====================================================================

import numpy as np
h = 1e-6
z = np.array([-2.0, -0.5, 0.5, 2.0])          # test points, away from kinks at 0
s = 1/(1+np.exp(-z))                           # sigmoid values

acts = {                                       # name: (function, analytic f'(z))
    "sigmoid":  (lambda t: 1/(1+np.exp(-t)),              s*(1-s)),
    "tanh":     (np.tanh,                                 1-np.tanh(z)**2),
    "softplus": (lambda t: np.log1p(np.exp(t)),           1/(1+np.exp(-z))),
    "swish":    (lambda t: t/(1+np.exp(-t)),              s + z*s*(1-s)),
    "elu(a=1)": (lambda t: np.where(t>0, t, np.exp(t)-1), np.where(z>0, 1.0, np.exp(z))),
}
print("activation   max|analytic - central difference|")
for name, (f, d_ana) in acts.items():
    d_num = (f(z+h) - f(z-h)) / (2*h)          # central difference
    print(f"{name:9}   {np.max(np.abs(d_ana - d_num)):.2e}")   # all ~1e-8: derivatives verified
