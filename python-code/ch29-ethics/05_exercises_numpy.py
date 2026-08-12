# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 29: Ethics, fairness & safety
# Section: Exercises
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "ethics"
# ====================================================================

import numpy as np
np.random.seed(0)
truth = (np.random.rand(100000) < 0.30).astype(int)   # 30% hold the sensitive trait
coin1 = np.random.rand(len(truth)) < 0.5              # heads: answer truthfully
coin2 = np.random.rand(len(truth)) < 0.5              # tails: answer a random yes/no
resp  = np.where(coin1, truth, coin2.astype(int))     # the released, noisy answers
p_yes = resp.mean()                                   # E[resp] = 0.5*trait + 0.25
print("fraction answering yes        =", round(float(p_yes), 4))
print("debiased estimate of the rate =", round(float(2*p_yes - 0.5), 4), " (true 0.30)")
print("likelihood ratio 0.75/0.25    =", 0.75/0.25, " -> eps = ln 3 =", round(float(np.log(3)), 3))
    
