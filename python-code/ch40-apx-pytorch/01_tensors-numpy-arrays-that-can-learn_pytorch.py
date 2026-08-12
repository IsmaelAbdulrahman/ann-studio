# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Tensors: NumPy arrays that can learn
# Code example 1 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import numpy as np

a = torch.tensor([[1., 2.], [3., 4.]])   # from a Python list (float32 by default)
b = torch.zeros(2, 2)
r = torch.randn(2, 2)                     # standard-normal random
print(a.shape, a.dtype)                   # torch.Size([2, 2]) torch.float32

# NumPy <-> tensor (shares memory on the CPU)
n = np.array([1., 2., 3.])
t = torch.from_numpy(n)
print("as tensor:", t, " back to numpy:", t.numpy())

# reshape, index and slice just like NumPy
print(a.view(4))          # 1-D view, shares storage (cf. a.reshape(4))
print(a[:, 0])            # first column -> tensor([1., 3.])

# pick a device once, move data/models onto it
device = "cuda" if torch.cuda.is_available() else "cpu"
x = a.to(device)
print("on device:", x.device)

print(a + b)        # elementwise, like NumPy
print(a @ a)        # matrix multiply
