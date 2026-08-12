# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Attention in code
# Code example 4 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import torch, torch.nn as nn
import torch.nn.functional as F

def attention(Q, K, V, mask=None):        # scaled dot-product attention
    d = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / d ** 0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    w = F.softmax(scores, dim=-1)
    return w @ V, w

Q = torch.tensor([[1., 0.]])
K = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
out, w = attention(Q, K, K)               # here values = keys
print(w)      # tensor([[0.4011, 0.1978, 0.4011]])
print(out)    # tensor([[0.8022, 0.5989]])

# the batched, multi-head path used in real transformers:
mha = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
x = torch.randn(2, 10, 64)                # (batch, sequence, features)
y, attn = mha(x, x, x)                    # self-attention: Q = K = V = x
print(y.shape)                            # torch.Size([2, 10, 64])
