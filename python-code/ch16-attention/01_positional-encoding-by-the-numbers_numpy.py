# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 16: Attention & the transformer
# Section: Positional encoding, by the numbers
# Code example 1 of 7 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "attention"
# ====================================================================

import numpy as np
np.set_printoptions(precision=3, suppress=True)

d = 4                              # model dimension (even)
positions = np.arange(4)           # tokens at positions 0..3
i = np.arange(d // 2)              # 0, 1
denom = 10000.0 ** (2 * i / d)     # one frequency per dimension pair

PE = np.zeros((len(positions), d))
for pos in positions:
    PE[pos, 0::2] = np.sin(pos / denom)   # even dims: sine
    PE[pos, 1::2] = np.cos(pos / denom)   # odd  dims: cosine

print("positional encodings (rows = positions 0..3):")
print(PE)
print("PE[0] . PE[1] =", round(float(PE[0] @ PE[1]), 4))
print("PE[0] . PE[3] =", round(float(PE[0] @ PE[3]), 4))
