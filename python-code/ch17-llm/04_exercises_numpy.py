# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 17: Language models & the transformer era
# Section: Exercises
# Code example 4 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "llm"
# ====================================================================

import numpy as np
np.random.seed(0)
logits = np.array([2.0, 1.0, 0.5, 0.0, -1.0])
def softmax(z):
    z = z - z.max(); e = np.exp(z); return e / e.sum()
p     = softmax(logits)                       # full next-token distribution
order = np.argsort(p)[::-1]                   # indices high -> low probability
cum   = np.cumsum(p[order])
keep  = order[:np.searchsorted(cum, 0.9) + 1] # smallest nucleus reaching 0.9
q     = np.zeros_like(p); q[keep] = p[keep]
q     = q / q.sum()                           # renormalise over the nucleus
print("full probs  :", np.round(p, 4))
print("nucleus ids :", np.sort(keep))         # [0 1 2 3]
print("nucleus prob:", np.round(q, 4))        # [0.5793 0.2131 0.1293 0.0784 0.]
    
