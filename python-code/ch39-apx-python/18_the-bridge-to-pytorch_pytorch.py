# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: The bridge to PyTorch
# Code example 18 of 22 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

import numpy as np, torch

x_np = np.array([[1., 2.], [3., 4.]])
x_t  = torch.from_numpy(x_np)          # a tensor sharing the array's memory
print(x_t, x_t.dtype)

y = x_t @ x_t                          # same @ operator as NumPy
print(y)
print("back to NumPy:", y.numpy())     # and back again
