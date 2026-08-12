# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 10: Initialization & the vanishing gradient
# Section: Initialization in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "init"
# ====================================================================

import torch, torch.nn as nn

layer = nn.Linear(256, 256)
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')   # He, for ReLU
nn.init.zeros_(layer.bias)

tanh_layer = nn.Linear(256, 256)
nn.init.xavier_normal_(tanh_layer.weight)                    # Xavier, for tanh

# initialize a whole network by walking its modules
net = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                    nn.Linear(256, 64),  nn.ReLU(),
                    nn.Linear(64, 10))
for m in net.modules():
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
