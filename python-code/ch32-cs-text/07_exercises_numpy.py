# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 32: Case study: text sentiment analysis
# Section: Exercises
# Code example 7 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-text"
# ====================================================================

import numpy as np
train = [("a great and brilliant movie i loved it",1),("wonderful acting and a great story",1),
 ("brilliant fun and truly wonderful",1),("i loved the great cast so good",1),
 ("a boring and terrible waste of time",0),("terrible dull and awful acting",0),
 ("so boring i hated the dull story",0),("awful terrible and a total waste",0)]
vocab = ["<unk>"] + sorted(set(w for s,_ in train for w in s.split()))   # id 0 = <unk>
idx   = {w:i for i,w in enumerate(vocab)}
def bow(s):
    v = np.zeros(len(vocab))
    for t in s.split():
        v[idx.get(t, 0)] += 1                     # unknown word -> <unk> bucket
    return v
X = np.array([bow(s) for s,_ in train]); y = np.array([l for _,l in train], float)
w = np.zeros(len(vocab)); b = 0.0
for _ in range(600):
    p = 1/(1+np.exp(-(X@w+b))); g = p - y
    w -= 0.3*(X.T@g)/len(y); b -= 0.3*g.mean()
s = "zorp glimf nonsense"                          # every word is out-of-vocabulary
v = bow(s); p = 1/(1+np.exp(-(v@w+b)))
print("all-unknown review -> <unk> count =", int(v[0]))
print(f"prediction = {p:.2f}   (w[<unk>] = {w[0]:+.2f}, bias b = {b:+.2f})")
