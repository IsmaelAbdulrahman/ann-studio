# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 6: Loss functions & maximum likelihood
# Section: The losses in code
# Code example 3 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "loss"
# ====================================================================

import numpy as np
def mse_grad(r):        return r
def mae_grad(r):        return np.sign(r)
def huber_grad(r, d=1.0):
    return np.where(np.abs(r) <= d, r, d*np.sign(r))

r = np.array([0.2, -0.5, 1.0, 8.0])   # residuals (yhat - y); last one is an outlier
print("residuals  :", r)
print("MSE   grad :", np.round(mse_grad(r), 2), " <- outlier dominates (8.0)")
print("MAE   grad :", np.round(mae_grad(r), 2), " <- all equal magnitude")
print("Huber grad :", np.round(huber_grad(r), 2), " <- outlier capped at delta=1")
