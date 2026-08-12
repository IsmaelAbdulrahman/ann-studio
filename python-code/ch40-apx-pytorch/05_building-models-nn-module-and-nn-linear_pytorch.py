# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Building models: nn.Module and nn.Linear
# Code example 5 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)   # weight (d_hidden, d_in) + bias
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))            # hidden layer + ReLU
        return self.fc2(x)                     # raw output "logits"

model = MLP(4, 16, 3)
print(model)
out = model(torch.randn(8, 4))                 # a batch of 8 -> shape (8, 3)
print("output shape:", out.shape)
print("param count :", sum(p.numel() for p in model.parameters()))  # 131
