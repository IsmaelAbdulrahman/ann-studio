# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 21: Generative models: GANs to diffusion
# Section: Generative models in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "generative"
# ====================================================================

import torch, torch.nn as nn
import torch.nn.functional as F

# ---- a tiny GAN: two networks, two optimizers, one alternating update ----
G = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 784), nn.Tanh())
D = nn.Sequential(nn.Linear(784, 128), nn.LeakyReLU(0.2), nn.Linear(128, 1))
optG = torch.optim.Adam(G.parameters(), 2e-4, betas=(0.5, 0.999))
optD = torch.optim.Adam(D.parameters(), 2e-4, betas=(0.5, 0.999))
bce = nn.BCEWithLogitsLoss()

real = torch.rand(32, 784)
z = torch.randn(32, 64); fake = G(z)
lossD = bce(D(real), torch.ones(32, 1)) + bce(D(fake.detach()), torch.zeros(32, 1))
optD.zero_grad(); lossD.backward(); optD.step()      # D: reals->1, fakes->0
lossG = bce(D(fake), torch.ones(32, 1))              # G: fool D into saying 1
optG.zero_grad(); lossG.backward(); optG.step()

# ---- one diffusion training step: predict the noise added at a random t ----
eps_net = nn.Sequential(nn.Linear(784 + 1, 256), nn.ReLU(), nn.Linear(256, 784))
x0 = torch.rand(32, 784); t = torch.randint(1, 1000, (32, 1)).float()
abar = torch.cos(t / 1000 * 1.5708) ** 2             # a valid schedule: 1 -> 0
eps = torch.randn_like(x0)
xt = abar.sqrt() * x0 + (1 - abar).sqrt() * eps      # forward jump to step t
loss = F.mse_loss(eps_net(torch.cat([xt, t / 1000], 1)), eps)
loss.backward()
