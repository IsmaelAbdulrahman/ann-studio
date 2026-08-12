# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 18: Graph neural networks
# Section: Graph networks in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "gnn"
# ====================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_adjacency(A):                  # A: dense (n, n) of 0/1
    A_tilde    = A + torch.eye(A.size(0))      # add self-loops:  A~ = A + I
    d          = A_tilde.sum(1)                # degrees
    d_inv_sqrt = torch.diag(d.pow(-0.5))       # D~^(-1/2)
    return d_inv_sqrt @ A_tilde @ d_inv_sqrt   # A^ = D~^-1/2 A~ D~^-1/2

class GCNLayer(nn.Module):
    """One Kipf-Welling graph-convolution layer (dense adjacency)."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)   # the shared W^(l)

    def forward(self, H, A_hat):               # H: (n, in);  A_hat: (n, n)
        return A_hat @ self.lin(H)             # A^ (H W)

class GCN(nn.Module):                          # a 2-layer node classifier
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        self.gc1 = GCNLayer(in_dim, hidden)
        self.gc2 = GCNLayer(hidden, n_classes)

    def forward(self, X, A_hat):
        H = F.relu(self.gc1(X, A_hat))         # hidden node embeddings
        return self.gc2(H, A_hat)              # per-node class logits

n, f = 4, 2
A     = torch.tensor([[0,1,1,0],[1,0,1,0],[1,1,0,1],[0,0,1,0]], dtype=torch.float)
A_hat = normalized_adjacency(A)                # precompute once, reuse every layer
model = GCN(f, 16, 3)
logits = model(torch.randn(n, f), A_hat)       # -> (4, 3): 3-class scores per node
print(logits.shape)                            # torch.Size([4, 3])
