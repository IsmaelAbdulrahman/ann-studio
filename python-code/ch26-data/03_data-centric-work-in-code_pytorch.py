# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 26: Data-centric deep learning
# Section: Data-centric work in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "data"
# ====================================================================

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

# --- 1. focal loss as a drop-in module (Lin et al., 2017) ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):        # alpha: optional per-class weights
        super().__init__()
        self.gamma, self.alpha = gamma, alpha
    def forward(self, logits, target):
        logp  = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)   # log p_t of the true class
        pt    = logpt.exp()                                  # p_t
        loss  = -(1 - pt) ** self.gamma * logpt              # FL = -(1-pt)^gamma * log pt
        if self.alpha is not None:
            loss = loss * self.alpha[target]                 # class-balanced variant
        return loss.mean()

# --- 2. WeightedRandomSampler: oversample the minority into every mini-batch ---
counts   = torch.bincount(train_labels)                # e.g. tensor([900, 100])
class_w  = 1.0 / counts.float()                        # inverse frequency
sample_w = class_w[train_labels]                       # one weight per training example
sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)
loader   = DataLoader(train_ds, batch_size=64, sampler=sampler)   # (no shuffle= with a sampler)

# --- 3. torchvision augmentation: label-preserving transforms only ---
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),   # position / scale invariance
    transforms.RandomHorizontalFlip(),                     # left-right invariance
    transforms.ColorJitter(0.4, 0.4, 0.4),                 # lighting invariance
    transforms.RandAugment(num_ops=2, magnitude=9),        # an automated augmentation policy
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],            # ImageNet stats (fit on train!)
                         [0.229, 0.224, 0.225]),
])
test_tf = transforms.Compose([                             # NO random ops at test time
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
