# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 11: Normalization: batch, layer & beyond
# Section: Normalization in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "normalization"
# ====================================================================

import torch, torch.nn as nn

x = torch.randn(32, 64)                       # batch of 32, 64 features

bn = nn.BatchNorm1d(64)                        # per-feature, statistics over the batch
ln = nn.LayerNorm(64)                          # per-example, statistics over the 64 features
img = torch.randn(8, 32, 16, 16)              # (N, C, H, W)
gn = nn.GroupNorm(num_groups=4, num_channels=32)   # per-example, per channel-group

bn.train(); y = bn(x)                          # TRAIN: use batch stats, update running mean/var
bn.eval();  y = bn(x)                          # EVAL:  use the frozen running mean/var (batch-free)
print(ln(x).shape, gn(img).shape)             # torch.Size([32, 64])  torch.Size([8, 32, 16, 16])

class RMSNorm(nn.Module):                      # as used in LLaMA / T5: no mean subtraction
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.g * x / rms                # scale by the root-mean-square only
print(RMSNorm(64)(x).shape)                    # torch.Size([32, 64])
