# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 32: Case study: text sentiment analysis
# Section: In code
# Code example 3 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-text"
# ====================================================================

import numpy as np

train = [                                       # (review, 1 = positive, 0 = negative)
 ("a great and brilliant movie i loved it", 1),
 ("wonderful acting and a great story", 1),
 ("brilliant fun and truly wonderful", 1),
 ("i loved the great cast so good", 1),
 ("a boring and terrible waste of time", 0),
 ("terrible dull and awful acting", 0),
 ("so boring i hated the dull story", 0),
 ("awful terrible and a total waste", 0),
]
vocab = sorted(set(w for s, _ in train for w in s.split()))
idx   = {w: i for i, w in enumerate(vocab)}
def bow(s):
    v = np.zeros(len(vocab))
    for w in s.split():
        if w in idx: v[idx[w]] += 1
    return v
X = np.array([bow(s) for s, _ in train])
y = np.array([lab for _, lab in train], dtype=float)

# logistic regression over word counts, trained from scratch
w = np.zeros(len(vocab)); b = 0.0
for ep in range(600):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    g = p - y                                   # binary cross-entropy gradient
    w -= 0.3 * (X.T @ g) / len(y)
    b -= 0.3 * g.mean()

print("predictions on unseen reviews:")
for s in ["a wonderful and brilliant story",
          "an awful and boring waste",
          "great acting but a boring story"]:
    p = 1 / (1 + np.exp(-(bow(s) @ w + b)))
    print(f"  {p:.2f}  {'POS' if p > 0.5 else 'NEG'}   |  {s}")

order = np.argsort(w)
print("\nmost negative words:", [vocab[i] for i in order[:3]])
print("most positive words:", [vocab[i] for i in order[-3:]][::-1])
