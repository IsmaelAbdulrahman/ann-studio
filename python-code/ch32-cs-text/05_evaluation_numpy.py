# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 32: Case study: text sentiment analysis
# Section: Evaluation
# Code example 5 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "cs-text"
# ====================================================================

import numpy as np

train = [                                        # (review, label)  1 = positive
 ("a great and brilliant movie i loved it", 1),("wonderful acting and a great story", 1),
 ("brilliant fun and truly wonderful", 1),("i loved the great cast so good", 1),
 ("a boring and terrible waste of time", 0),("terrible dull and awful acting", 0),
 ("so boring i hated the dull story", 0),("awful terrible and a total waste", 0)]
test = [                                         # held-out reviews with human labels
 ("a wonderful and brilliant story", 1),("i loved the great fun cast", 1),
 ("great acting and a good movie", 1),("truly wonderful and so good", 1),
 ("an awful and boring waste", 0),("terrible dull and awful acting", 0),
 ("not great and not good", 0),("great story but truly terrible acting", 0)]

vocab = sorted(set(w for s, _ in train for w in s.split()))
idx   = {w: i for i, w in enumerate(vocab)}
def bow(s):
    v = np.zeros(len(vocab))
    for w in s.split():
        if w in idx: v[idx[w]] += 1
    return v
X = np.array([bow(s) for s, _ in train]); y = np.array([l for _, l in train], float)

w = np.zeros(len(vocab)); b = 0.0                # logistic regression, from scratch
for _ in range(600):
    p = 1/(1+np.exp(-(X@w+b))); g = p - y
    w -= 0.3*(X.T@g)/len(y); b -= 0.3*g.mean()

yt   = np.array([l for _, l in test])
prob = np.array([1/(1+np.exp(-(bow(s)@w+b))) for s, _ in test])
pred = (prob > 0.5).astype(int)

TP = int(((pred==1)&(yt==1)).sum()); TN = int(((pred==0)&(yt==0)).sum())
FP = int(((pred==1)&(yt==0)).sum()); FN = int(((pred==0)&(yt==1)).sum())
acc = (pred==yt).mean(); prec = TP/(TP+FP); rec = TP/(TP+FN)
f1  = 2*prec*rec/(prec+rec)
print("confusion    pred NEG  pred POS")
print(f" actual NEG     {TN}         {FP}")
print(f" actual POS     {FN}         {TP}")
print(f"accuracy {acc:.2f}  precision {prec:.2f}  recall {rec:.2f}  F1 {f1:.2f}\n")
for (s, _), t, pp, pr in zip(test, yt, pred, prob):
    print(f"  {pr:.2f} {'POS' if pp else 'NEG'} (true {'POS' if t else 'NEG'}) {'MISS' if t!=pp else '    '} | {s}")
