# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Image datasets: MNIST, Fashion-MNIST, CIFAR-10
# Code example 1 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

tfm = transforms.Compose([
    transforms.ToTensor(),                       # uint8 [0,255] -> float [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean / std
])

train = datasets.MNIST(root="data", train=True,  download=True, transform=tfm)
test  = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

loader = DataLoader(train, batch_size=64, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)   # torch.Size([64, 1, 28, 28]) torch.Size([64])
# swap datasets.FashionMNIST or datasets.CIFAR10 in for the other two
