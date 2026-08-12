# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 5: Features, embeddings & what a network learns
# Section: Representations in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "representations"
# ====================================================================

import torch, torch.nn as nn
import torch.nn.functional as F

emb = nn.Embedding(num_embeddings=4, embedding_dim=3)      # a learnable V x d table
with torch.no_grad():                                      # plant the chapter's E
    emb.weight.copy_(torch.tensor([[3., 0., 1.],
                                   [3., 0., 2.],
                                   [0., 3., 2.],
                                   [0., 3., 3.]]))

idx = torch.tensor([1, 0, 2])                              # dog, cat, car
gathered = emb(idx)                                        # (3, 3) differentiable lookup
onehot   = F.one_hot(idx, num_classes=4).float()           # (3, 4)
print(torch.allclose(gathered, onehot @ emb.weight))       # True: gather == matmul

cos = F.cosine_similarity(gathered[:, None], gathered[None, :], dim=-1)
print(cos)                    # cat-dog high, car far; only used rows get gradients
