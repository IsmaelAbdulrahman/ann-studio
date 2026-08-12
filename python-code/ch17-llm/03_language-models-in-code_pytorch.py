# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 17: Language models & the transformer era
# Section: Language models in code
# Code example 3 of 4 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "llm"
# ====================================================================

import torch, torch.nn as nn
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.qkv  = nn.Linear(d_model, 3 * d_model)   # fused Q, K, V projection
        self.proj = nn.Linear(d_model, d_model)       # output mix  W_O

    def forward(self, x):                             # x: (B, T, d_model)
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head                          # per-head dimension
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)   # (B, h, T, hd)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att  = (q @ k.transpose(-2, -1)) / hd ** 0.5   # (B, h, T, T) scores
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        att  = att.masked_fill(mask, float('-inf'))    # forbid attending ahead
        att  = F.softmax(att, dim=-1)
        y = att @ v                                    # (B, h, T, hd)
        y = y.transpose(1, 2).reshape(B, T, C)         # concat heads
        return self.proj(y)

@torch.no_grad()
def generate(model, idx, max_new, temperature=1.0, greedy=False):
    # model(idx) returns logits of shape (B, T, vocab); idx is (B, T) token ids
    for _ in range(max_new):
        logits = model(idx)[:, -1, :] / temperature    # next-token logits (B, V)
        if greedy:
            nxt = logits.argmax(dim=-1, keepdim=True)   # deterministic argmax
        else:
            probs = F.softmax(logits, dim=-1)
            nxt   = torch.multinomial(probs, num_samples=1)   # temperature sample
        idx = torch.cat([idx, nxt], dim=1)              # append and feed back in
    return idx                                          # add top-k/top-p to taste
