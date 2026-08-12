# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: The eight activations in code
# Code example 3 of 9 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import numpy as np
from math import erf
erf_v = np.vectorize(erf)                      # standard-normal helper for GELU

def sigmoid(z):  return 1/(1+np.exp(-z))
def tanh(z):     return np.tanh(z)
def relu(z):     return np.maximum(0, z)
def leaky(z, a=0.01): return np.where(z>0, z, a*z)
def elu(z, a=1.0):    return np.where(z>0, z, a*(np.exp(z)-1))
def softplus(z): return np.log1p(np.exp(z))
def gelu(z):     return z*0.5*(1+erf_v(z/np.sqrt(2)))   # exact GELU = z*Phi(z)
def swish(z):    return z*sigmoid(z)

z = np.array([-2.0, 0.0, 2.0])
acts = [("sigmoid",sigmoid),("tanh",tanh),("relu",relu),("leaky",leaky),
        ("elu",elu),("softplus",softplus),("gelu",gelu),("swish",swish)]
print("            z = -2      0     +2")
for name, f in acts:
    v = np.round(f(z), 3)
    print(f"{name:>9}: {v[0]:7.3f} {v[1]:6.3f} {v[2]:6.3f}")
