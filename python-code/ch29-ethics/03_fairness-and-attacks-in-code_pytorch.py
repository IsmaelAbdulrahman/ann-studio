# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 29: Ethics, fairness & safety
# Section: Fairness and attacks in code
# Code example 3 of 6 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "ethics"
# ====================================================================

import torch, torch.nn.functional as F

# a trained classifier `model` and one correctly-classified example (x, label)
model.eval()
x = x.clone().detach().requires_grad_(True)   # image tensor (1, C, H, W), pixels in [0,1]
y = torch.tensor([label])                     # the true class index

logits = model(x)
loss   = F.cross_entropy(logits, y)           # the loss of Chapter 6
model.zero_grad()
loss.backward()                               # fills x.grad = d loss / d x

eps   = 8/255                                  # a small L-infinity budget
x_adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()   # FGSM step, kept a valid image

with torch.no_grad():
    p_clean = model(x).softmax(1)
    p_adv   = model(x_adv).softmax(1)
print("clean:", p_clean.argmax(1).item(), "conf", round(p_clean.max().item(), 3))
print("adv  :", p_adv.argmax(1).item(),   "conf", round(p_adv.max().item(),   3))
# same tiny eps typically flips the label to a confident wrong class (Goodfellow et al. 2015)
