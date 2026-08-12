# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 32: Case study: text sentiment analysis
# Section: A ladder of representations
# Code example 2 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-text"
# ====================================================================

import numpy as np
np.random.seed(0)

train = [                                        # (review, label)  1 = positive
 ("a great and brilliant movie i loved it", 1),("wonderful acting and a great story", 1),
 ("brilliant fun and truly wonderful", 1),("i loved the great cast so good", 1),
 ("a boring and terrible waste of time", 0),("terrible dull and awful acting", 0),
 ("so boring i hated the dull story", 0),("awful terrible and a total waste", 0)]
vocab = sorted(set(w for s, _ in train for w in s.split()))
idx   = {w: i for i, w in enumerate(vocab)}
y     = np.array([l for _, l in train], float)

dim = 3
E   = 0.01*np.random.randn(len(vocab), dim)      # embedding table  (V x dim)
w   = np.zeros(dim); b = 0.0                      # logistic layer on the mean vector

def meanvec(s):
    ids = [idx[t] for t in s.split() if t in idx]
    return E[ids].mean(0), ids                    # mean-pool the word embeddings

for ep in range(4000):
    gE = np.zeros_like(E); gw = np.zeros(dim); gb = 0.0
    for (s, _), t in zip(train, y):
        r, ids = meanvec(s)
        d = 1/(1+np.exp(-(r@w+b))) - t             # binary cross-entropy gradient
        gw += d*r; gb += d
        for j in ids: gE[j] += d*w/len(ids)        # split the mean's gradient
    w -= 0.5*gw/len(y); b -= 0.5*gb/len(y); E -= 0.5*gE/len(y)

pred = np.array([1 if meanvec(s)[0]@w + b > 0 else 0 for s, _ in train])
print("averaged-embedding train accuracy =", (pred == y).mean())   # expected: 1.0
