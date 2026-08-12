# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Datasets and DataLoader
# Code example 10 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# quickest path: wrap tensors you already have
X = torch.randn(500, 4)
y = torch.randint(0, 3, (500,))
train_ds = TensorDataset(X, y)

# or define your own for custom / lazy loading
class MyData(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

loader = DataLoader(train_ds, batch_size=32, shuffle=True)
xb, yb = next(iter(loader))
print(xb.shape, yb.shape)     # torch.Size([32, 4]) torch.Size([32])
