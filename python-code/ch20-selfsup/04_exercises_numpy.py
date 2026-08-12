# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 20: Self-supervised & contrastive learning
# Section: Exercises
# Code example 4 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "selfsup"
# ====================================================================

import numpy as np
tau = 0.1
def infonce(sim_pos, sim_negs):            # one positive, a list of negatives
    num = np.exp(sim_pos / tau)
    den = num + np.sum(np.exp(np.array(sim_negs) / tau))
    return -np.log(num / den)

print("base  pos=0.5 neg=0.4 :", round(float(infonce(0.5, [0.4])), 4))   # 0.3133
print("raise pos=0.7 neg=0.4 :", round(float(infonce(0.7, [0.4])), 4))   # 0.0486
print("lower pos=0.5 neg=0.2 :", round(float(infonce(0.5, [0.2])), 4))   # 0.0486
