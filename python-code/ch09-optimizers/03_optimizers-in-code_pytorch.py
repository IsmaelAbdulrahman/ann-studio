# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 9: Momentum, RMSProp & Adam
# Section: Optimizers in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "optimizers"
# ====================================================================

import torch

w = torch.tensor([1.0, 1.0], requires_grad=True)
A = torch.tensor([20.0, 1.0])                 # ravine curvatures
def loss(w): return 0.5 * (A * w * w).sum()

opt = torch.optim.Adam([w], lr=0.1)           # try SGD([w], lr=0.02, momentum=0.9)
for step in range(200):
    opt.zero_grad()
    loss(w).backward()                        # autograd fills w.grad
    opt.step()                                # one update rule applied
print(w.detach())                             # -> close to [0., 0.]

# AdamW: decoupled weight decay is a drop-in replacement
opt = torch.optim.AdamW([w], lr=0.1, weight_decay=0.01)
