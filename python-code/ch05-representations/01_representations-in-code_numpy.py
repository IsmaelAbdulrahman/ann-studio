# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 5: Features, embeddings & what a network learns
# Section: Representations in code
# Code example 1 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "representations"
# ====================================================================

import numpy as np
np.random.seed(0)

# 4-word vocabulary, 3-dimensional embedding matrix E (rows are word vectors)
words = ["cat", "dog", "car", "van"]
E = np.array([[3., 0., 1.],      # cat
              [3., 0., 2.],      # dog
              [0., 3., 2.],      # car
              [0., 3., 3.]])     # van

# a one-hot vector times E selects a row -- lookup IS a matrix multiply
onehot_dog = np.array([0., 1., 0., 0.])          # index 1 = "dog"
print("one-hot . E =", onehot_dog @ E, " == row 1 =", E[1])

# cosine-similarity matrix over all four word vectors
unit = E / np.linalg.norm(E, axis=1, keepdims=True)
C = unit @ unit.T                                 # 4x4 cosine similarities
print("cosine similarity matrix:")
print(np.round(C, 3))

# which word is closest to "cat" (ignoring itself)?
i = words.index("cat")
sims = C[i].copy(); sims[i] = -np.inf
print("closest to cat:", words[int(np.argmax(sims))],
      " cos =", round(float(sims.max()), 3))   # expected: dog  cos = 0.965
