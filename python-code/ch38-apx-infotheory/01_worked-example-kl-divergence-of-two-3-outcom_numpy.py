# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 38: Appendix D · Information theory
# Section: KL divergence: the price of the wrong model → Worked example — KL divergence of two 3-outcome distributions, and the identity
# Code example 1 of 2 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-infotheory"
# ====================================================================

import numpy as np
np.random.seed(0)
EPS = 1e-12                                  # guards log(0); p=0 -> 0*log(EPS)=0

def entropy(p, bits=True):
    p  = np.asarray(p, dtype=float)
    lg = np.log2 if bits else np.log         # log2 -> bits, log -> nats
    return float(-np.sum(p * lg(p + EPS)))

def cross_entropy(p, q, bits=True):
    p, q = np.asarray(p, float), np.asarray(q, float)
    lg = np.log2 if bits else np.log
    return float(-np.sum(p * lg(q + EPS)))

def kl(p, q, bits=True):
    p, q = np.asarray(p, float), np.asarray(q, float)
    lg = np.log2 if bits else np.log
    return float(np.sum(p * lg((p + EPS) / (q + EPS))))

p = np.array([0.5,  0.25, 0.25])             # true distribution over 3 outcomes
q = np.array([0.25, 0.25, 0.5 ])             # a model of the same 3 outcomes

H, Hpq, D = entropy(p), cross_entropy(p, q), kl(p, q)
print("H(p)      =", round(H, 4),   "bits =", round(entropy(p, bits=False), 4), "nats")
print("H(p,q)    =", round(Hpq, 4), "bits")
print("KL(p||q)  =", round(D, 4),   "bits")
assert np.isclose(Hpq, H + D)                # cross-entropy = entropy + KL divergence
print("identity  H(p,q) == H(p) + KL(p||q):", bool(np.isclose(Hpq, H + D)))

r = np.array([0.1, 0.2, 0.7])                # asymmetry: KL(p||r) != KL(r||p)
print("asymmetry KL(p||r) =", round(kl(p, r), 4), " vs  KL(r||p) =", round(kl(r, p), 4))
