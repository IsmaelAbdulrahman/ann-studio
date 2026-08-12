# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: The training loop
# Code example 8 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import torch.nn as nn

def train(model, loader, epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()                 # swap for your task
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()                               # training mode
        running = 0.0
        for xb, yb in loader:                       # one batch at a time
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()                         # 1. clear old gradients
            pred = model(xb)                        # 2. forward pass
            loss = loss_fn(pred, yb)                # 3. compute loss
            loss.backward()                         # 4. backprop (autograd)
            opt.step()                              # 5. update the weights
            running += loss.item() * xb.size(0)
        avg = running / len(loader.dataset)
        print(f"epoch {epoch+1:2d}  loss {avg:.4f}")
    return model
