# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 1: From biological to artificial neurons
# Section: The neuron in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "neuron"
# ====================================================================

import torch, torch.nn as nn

neuron = nn.Linear(3, 1)                  # 3 inputs -> 1 output (a weight row + bias)
x = torch.tensor([4., 2., 1.])
a = torch.sigmoid(neuron(x))              # z = Wx + b, then sigmoid
print(a)
