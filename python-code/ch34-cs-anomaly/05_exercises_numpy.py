# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 34: Case study: anomaly detection with autoencoders
# Section: Exercises
# Code example 5 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-anomaly"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
D, d = 8, 2; A = rng.randn(d, D)
def normal(n): return rng.randn(n, d) @ A + 0.10*rng.randn(n, D)
Xtr = normal(500); mu, sd = Xtr.mean(0), Xtr.std(0); z = lambda X: (X - mu)/sd
W1 = rng.randn(D, d)*0.1; W2 = rng.randn(d, D)*0.1; S = z(Xtr)
for ep in range(4000):
    H = S@W1; G = 2*(H@W2 - S)/len(S); W2 -= 0.05*(H.T@G); W1 -= 0.05*(S.T@(G@W2.T))
broken = 5                                    # <- try 0..7
x = normal(1).copy(); x[0, broken] += 8.0     # one sensor spikes off the manifold
per = ((z(x)@W1@W2) - z(x))[0]**2             # squared error, per sensor (no averaging)
print("per-sensor error:", np.round(per, 2))
print("flagged sensor  :", int(per.argmax()), " (we broke sensor", broken, ")")
