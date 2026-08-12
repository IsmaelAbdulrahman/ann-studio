# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 13: Convolutional neural networks
# Section: Convolution in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "cnn"
# ====================================================================

import torch, torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),   # 28x28 -> 28x28
            nn.MaxPool2d(2),                                         #       -> 14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),  #       -> 14x14
            nn.MaxPool2d(2))                                        #       -> 7x7
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32*7*7, n_classes))
    def forward(self, x):
        return self.head(self.features(x))

net = SmallCNN()
x = torch.randn(8, 1, 28, 28)          # a batch of 8 grayscale 28x28 images
print(net(x).shape)                     # torch.Size([8, 10]) — one score per class
