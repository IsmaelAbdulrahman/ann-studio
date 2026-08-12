# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Autograd: gradients for free
# Code example 2 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch

# a leaf tensor we want gradients for
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x                 # PyTorch records this computation
y.backward()                   # compute dy/dx, store it in x.grad
print("y     :", y.item())     # 2^2 + 3*2 = 10
print("dy/dx :", x.grad.item())  # 2x + 3 = 7 at x = 2

# gradients ACCUMULATE across .backward() calls, so reset between steps
x.grad.zero_()
z = 5 * x
z.backward()
print("dz/dx :", x.grad.item())  # 5
