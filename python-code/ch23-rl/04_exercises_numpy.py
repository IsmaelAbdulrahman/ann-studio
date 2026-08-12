# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 23: Deep reinforcement learning
# Section: Exercises
# Code example 4 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rl"
# ====================================================================

import numpy as np
np.random.seed(0)
N, GOAL = 6, 5
def step(s, a):
    s2 = min(s + 1, N - 1) if a == 1 else max(s - 1, 0)
    r  = 1.0 if s2 == GOAL else -0.05          # step penalty on non-goal moves
    return s2, r, (s2 == GOAL)
Q = np.zeros((N, 2)); alpha, gamma, eps = 0.5, 0.9, 0.1
for ep in range(600):
    s = np.random.randint(0, GOAL)
    for t in range(50):
        a = np.random.randint(2) if np.random.rand() < eps else int(np.argmax(Q[s]))
        s2, r, done = step(s, a)
        Q[s, a] += alpha * (r + (0.0 if done else gamma * Q[s2].max()) - Q[s, a])
        s = s2
        if done: break
print("Q(s,right) =", np.round(Q[:GOAL, 1], 3))
print("greedy path:", "".join("R" if a == 1 else "L" for a in np.argmax(Q, 1)[:GOAL]))  # RRRRR
    
