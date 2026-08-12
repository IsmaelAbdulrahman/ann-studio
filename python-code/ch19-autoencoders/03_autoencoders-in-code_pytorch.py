# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 19: Autoencoders & representation learning
# Section: Autoencoders in code
# Code example 3 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "autoencoders"
# ====================================================================

import torch, torch.nn as nn
import torch.nn.functional as F

class AE(nn.Module):                       # plain autoencoder
    def __init__(self, n=784, d=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 128), nn.ReLU(), nn.Linear(128, d))
        self.dec = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, n))
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z

model = AE(); opt = torch.optim.Adam(model.parameters(), 1e-3)
x = torch.rand(64, 784)
xh, z = model(x)
loss = F.mse_loss(xh, x)                   # reconstruction loss
opt.zero_grad(); loss.backward(); opt.step()

class VAE(nn.Module):                       # variational autoencoder
    def __init__(self, n=784, d=32):
        super().__init__()
        self.enc = nn.Linear(n, 2 * d); self.dec = nn.Linear(d, n)
    def forward(self, x):
        mu, logvar = self.enc(x).chunk(2, dim=1)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterize
        kl = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=1).mean()
        return self.dec(z), kl
