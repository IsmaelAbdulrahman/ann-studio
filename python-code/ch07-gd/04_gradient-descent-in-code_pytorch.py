# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 7: Gradient descent & its variants
# Section: Gradient descent in code
# Code example 4 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "gd"
# ====================================================================

import torch

theta = torch.tensor([0.5], requires_grad=True)   # start point
opt = torch.optim.SGD([theta], lr=0.1)            # plain gradient descent

for step in range(30):
    opt.zero_grad()                                # clear last step's gradient
    loss = theta**4 - 3*theta**2 + 2               # the double-well loss
    loss.backward()                                # autograd fills theta.grad
    opt.step()                                      # theta <- theta - lr * grad

print(theta.item(), loss.item())      # ~1.2247, ~-0.25
# add momentum=0.9 to accelerate down ravines; see chapter 9
