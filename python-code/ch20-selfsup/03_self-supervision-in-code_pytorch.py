# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 20: Self-supervised & contrastive learning
# Section: Self-supervision in code
# Code example 3 of 4 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "selfsup"
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

def nt_xent(z1, z2, tau=0.5):
    # z1, z2: (N, d) projection-head outputs for the two augmented views
    N = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2N, d) unit vectors
    sim = (z @ z.t()) / tau                               # (2N, 2N) cosine / temperature
    sim.fill_diagonal_(float('-inf'))                     # forbid an example matching itself
    targets = (torch.arange(2 * N) + N) % (2 * N)         # positive of i is i+N (mod 2N)
    return F.cross_entropy(sim, targets)                  # InfoNCE = CE over candidates

z1 = torch.randn(256, 128)          # g(f(view1)) for a batch of 256 images
z2 = torch.randn(256, 128)          # g(f(view2))
loss = nt_xent(z1, z2, tau=0.5)
print(loss.item())                  # a single scalar; ~5.5 at random init, falls during training

# one optimizer step would then do:  loss.backward(); opt.step()
