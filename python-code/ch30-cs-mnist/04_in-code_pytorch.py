# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 30: Case study: handwritten-digit recognition (MNIST)
# Section: In code
# Code example 4 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "cs-mnist"
# ====================================================================

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))])   # MNIST mean/std
train = datasets.MNIST("./data", train=True,  download=True, transform=tf)
test  = datasets.MNIST("./data", train=False, download=True, transform=tf)
train_dl = DataLoader(train, batch_size=128, shuffle=True)
test_dl  = DataLoader(test,  batch_size=512)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, 3, padding=1)      # 28x28 -> 28x28
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)       # -> 14x14
        x = F.max_pool2d(F.relu(self.c2(x)), 2)       # -> 7x7
        return self.fc(x.flatten(1))                  # logits over 10 classes

dev = "cuda" if torch.cuda.is_available() else "cpu"
net = CNN().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(3):
    net.train()
    for xb, yb in train_dl:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        F.cross_entropy(net(xb), yb).backward()       # softmax + CE in one call
        opt.step()
    net.eval(); correct = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            correct += (net(xb.to(dev)).argmax(1).cpu() == yb).sum().item()
    print(f"epoch {epoch}   test acc {correct/len(test):.4f}")
