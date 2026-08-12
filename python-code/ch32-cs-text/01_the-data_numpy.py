# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 32: Case study: text sentiment analysis
# Section: The data
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-text"
# ====================================================================

import numpy as np

reviews = [
 "a great and brilliant movie i loved it",
 "a boring and terrible waste of time",
]
# tokenize (lowercase split) and build a vocabulary over the corpus
tokens = [r.split() for r in reviews]
vocab  = sorted(set(w for t in tokens for w in t))
idx    = {w: i for i, w in enumerate(vocab)}
print(f"vocabulary ({len(vocab)} words): {vocab}\n")

s   = reviews[0]
ids = [idx[w] for w in s.split()]
print(f"review    : {s}")
print(f"token ids : {ids}")

bow = np.zeros(len(vocab))                 # bag-of-words: count each vocab word
for w in s.split():
    bow[idx[w]] += 1
print(f"bag-of-words vector : {bow.astype(int)}")
