# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Exercises
# Code example 15 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
a = torch.zeros(4, 4)     # float32 by default (NumPy would give float64)
s = a.sum(dim=0)          # axis -> dim
print(s.shape, a.dtype)   # torch.Size([4]) torch.float32
