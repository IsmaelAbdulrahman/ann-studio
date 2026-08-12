# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 14: Modern CNN architectures & computer vision
# Section: Residual blocks in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "vision"
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """ResNet-18/34 residual block:  out = ReLU( branch(x) + shortcut(x) )."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        # First conv may downsample (stride) and change the channel count.
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

        # Identity skip when shapes agree; otherwise a 1x1 projection shortcut.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))

        nn.init.zeros_(self.bn2.weight)          # F(x)=0 at init -> starts as identity

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))    # 3x3 -> BN -> ReLU
        out = self.bn2(self.conv2(out))          # 3x3 -> BN
        out = out + self.shortcut(x)             # residual add (identity or projection)
        return F.relu(out)                       # ReLU AFTER the addition

# block = BasicBlock(64, 64); x = torch.randn(8, 64, 56, 56)
# block(x).shape  ->  torch.Size([8, 64, 56, 56])   # shape preserved
