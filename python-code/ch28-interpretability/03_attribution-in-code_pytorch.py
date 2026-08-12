# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 28: Interpretability & explainability
# Section: Attribution in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "interpretability"
# ====================================================================

import torch, torch.nn as nn, torch.nn.functional as F

# a small CNN; in practice this is any trained classifier (e.g. a torchvision ResNet)
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),        # <-- last conv: explain THIS
    nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 10))
model.eval()
target = model[3]                       # the second Conv2d, our feature extractor

feats, grads = {}, {}                    # hooks stash the tensors we need
target.register_forward_hook(       lambda m, i, o: feats.__setitem__("A",  o.detach()))
target.register_full_backward_hook( lambda m, gi, go: grads.__setitem__("dA", go[0].detach()))

x = torch.randn(1, 3, 32, 32)           # one input image, shape (1, 3, 32, 32)
logits = model(x)                        # forward pass fills feats["A"] -> (1,32,16,16)
cls = logits.argmax(1)                   # the predicted class
model.zero_grad()
logits[0, cls].backward()                # backprop the class score -> fills grads["dA"]

A     = feats["A"]                       # (1, 32, 16, 16) activations of last conv (Ch.13)
dA    = grads["dA"]                      # (1, 32, 16, 16) gradients d(score)/dA (Ch.8)
alpha = dA.mean(dim=(2, 3), keepdim=True)          # (1,32,1,1) importance weight per map
cam   = F.relu((alpha * A).sum(1, keepdim=True))   # (1,1,16,16) weighted sum, keep +ve
cam   = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
cam   = (cam - cam.amin()) / (cam.amax() - cam.amin() + 1e-8)   # normalise to [0,1]
print(cam.shape)                         # torch.Size([1, 1, 32, 32]) — a heatmap over x
