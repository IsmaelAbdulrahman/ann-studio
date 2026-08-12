# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Autograd: gradients for free
# Code example 4 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch

x = torch.randn(3, requires_grad=True)

# no_grad: a whole block where nothing is recorded (inference / evaluation)
with torch.no_grad():
    y = x * 2
print("inside no_grad, tracked? ", y.requires_grad)   # False

# detach: pull one tensor off the tape, leave the rest tracking
z = (x * 2).detach()
print("detached, tracked?      ", z.requires_grad)    # False
print("safe hand-off to numpy  :", z.numpy())         # no graph attached
