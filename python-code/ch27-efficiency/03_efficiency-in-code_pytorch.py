# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 27: Efficiency & deployment
# Section: Efficiency in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "efficiency"
# ====================================================================

import torch, torch.nn as nn
import torch.nn.utils.prune as prune

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))

# (1) unstructured magnitude pruning: zero the smallest 40% of each layer's weights
for m in model:
    if isinstance(m, nn.Linear):
        prune.l1_unstructured(m, name="weight", amount=0.4)   # add a 0/1 mask
        prune.remove(m, "weight")                             # bake the zeros in
# structured alternative — drop whole output rows (neurons):
# prune.ln_structured(m, name="weight", amount=0.25, n=2, dim=0)

# (2) post-training dynamic quantization: fp32 -> int8 weights for the Linear layers
qmodel = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8)   # activations quantized on the fly

x = torch.randn(1, 784)
print(qmodel(x).shape)          # torch.Size([1, 10]) -- same API, int8 matmuls, ~4x smaller
