# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 2: The perceptron & linear separability
# Section: The perceptron in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "perceptron"
# ====================================================================

import torch, torch.nn as nn

layer = nn.Linear(2, 1)                     # one linear unit: z = w·x + b
X = torch.tensor([[2., 2.], [1., 3.], [-1., -2.], [-2., -1.]])
y = torch.tensor([1., 1., 0., 0.])
eta = 1.0
for _ in range(10):                         # sweeps over the data
    for xi, yi in zip(X, y):
        z = layer(xi)
        yhat = (z >= 0).float()
        err = yi - yhat                     # in {-1, 0, +1}
        with torch.no_grad():               # the classic rule is not autograd
            layer.weight += eta * err * xi
            layer.bias   += eta * err
print("w =", layer.weight.data, " b =", layer.bias.data)
