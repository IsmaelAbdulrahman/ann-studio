# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 17: Language models & the transformer era
# Section: Language models in code
# Code example 1 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "llm"
# ====================================================================

import numpy as np
np.random.seed(0)

logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])     # scores over a 5-token vocab
vocab  = ["the", "cat", "sat", "on", "mat"]

def softmax(z):
    z = z - z.max()                                # subtract max for stability
    e = np.exp(z)
    return e / e.sum()

T = 0.7                                            # temperature below 1 sharpens
k = 3                                             # keep only the top-3 logits
scaled = logits / T
top    = np.argsort(scaled)[-k:]                  # indices of the k largest
filt   = np.full_like(scaled, -np.inf)            # everything else -> -inf
filt[top] = scaled[top]
probs  = softmax(filt)                            # renormalised over the top-k

print("top-k probs:", np.round(probs, 4))         # [0.7369 0.1766 0.0865 0. 0.]
tok = np.random.choice(len(vocab), p=probs)       # sample one token id
print("sampled id :", tok, "->", vocab[tok])       # 0 -> the
