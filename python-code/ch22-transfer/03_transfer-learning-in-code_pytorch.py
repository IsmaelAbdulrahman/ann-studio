# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 22: Transfer learning & fine-tuning
# Section: Transfer learning in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "transfer"
# ====================================================================

import torch
import torch.nn as nn
import torchvision.models as models

# ---- 1. Feature extraction: freeze the backbone, train a NEW head ----
net = models.resnet50(weights="IMAGENET1K_V2")     # pretrained backbone
for p in net.parameters():
    p.requires_grad = False                        # freeze every backbone weight
net.fc = nn.Linear(net.fc.in_features, 10)         # fresh head: 2048 -> 10 classes
# only net.fc.weight and net.fc.bias now have requires_grad = True
opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, net.parameters()), lr=1e-3)

# ---- 2. Fine-tuning with DISCRIMINATIVE (layer-wise) learning rates ----
opt = torch.optim.Adam([
    {"params": net.layer1.parameters(), "lr": 1e-5},   # early / general: tiny LR
    {"params": net.layer4.parameters(), "lr": 1e-4},   # late  / specific: larger
    {"params": net.fc.parameters(),     "lr": 1e-3},   # new head: largest
])

# ---- 3. A minimal LoRA-adapted linear layer ----
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base                                # pretrained Linear, frozen
        self.base.weight.requires_grad = False
        if base.bias is not None:
            self.base.bias.requires_grad = False
        d_in, d_out = base.in_features, base.out_features
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)  # down-projection  (r x d_in)
        self.B = nn.Parameter(torch.zeros(d_out, r))        # up-projection init 0 (d_out x r)
        self.scale = alpha / r
    def forward(self, x):                               # W x  +  (alpha/r) B A x
        return self.base(x) + self.scale * ((x @ self.A.t()) @ self.B.t())

layer = LoRALinear(nn.Linear(1024, 1024), r=8, alpha=16)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
print(trainable)          # 16384  =  2 * 1024 * 8   (the frozen 1,048,576 weights train nothing)
