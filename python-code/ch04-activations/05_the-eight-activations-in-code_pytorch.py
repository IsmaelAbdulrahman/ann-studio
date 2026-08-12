# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 4: Activation functions
# Section: The eight activations in code
# Code example 5 of 9 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "activations"
# ====================================================================

import torch
import torch.nn.functional as F

z = torch.tensor([-2.0, 0.0, 2.0])
print(torch.sigmoid(z))          # or z.sigmoid()
print(torch.tanh(z))
print(F.relu(z))
print(F.leaky_relu(z, 0.01))
print(F.elu(z, alpha=1.0))
print(F.softplus(z))
print(F.gelu(z))                 # exact by default; approximate='tanh' also available
print(F.silu(z))                 # SiLU is the same function as Swish: z * sigmoid(z)

logits = torch.tensor([2.0, 1.0, 0.1])
print(F.softmax(logits, dim=0))          # temperature: F.softmax(logits / T, dim=0)
