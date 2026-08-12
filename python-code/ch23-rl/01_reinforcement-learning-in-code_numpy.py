# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 23: Deep reinforcement learning
# Section: Reinforcement learning in code
# Code example 1 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rl"
# ====================================================================

import numpy as np
np.random.seed(0)

# ---- a 6-state chain (corridor). State 5 is the goal (terminal). ----
# actions: 0 = left, 1 = right.  Reward +1 only on stepping INTO the goal.
N, GOAL = 6, 5
def step(s, a):                                   # deterministic dynamics
    s2 = min(s + 1, N - 1) if a == 1 else max(s - 1, 0)
    r  = 1.0 if s2 == GOAL else 0.0
    return s2, r, (s2 == GOAL)

Q = np.zeros((N, 2))                              # value table, one row per state
alpha, gamma, eps = 0.5, 0.9, 0.1
for episode in range(400):
    s = np.random.randint(0, GOAL)                # random non-goal start state
    for t in range(50):                           # cap the episode length
        a = np.random.randint(2) if np.random.rand() < eps else int(np.argmax(Q[s]))
        s2, r, done = step(s, a)
        target = r + (0.0 if done else gamma * Q[s2].max())   # TD target
        Q[s, a] += alpha * (target - Q[s, a])     # tabular Q-learning update
        s = s2
        if done: break

opt = 0.9 ** (4 - np.arange(GOAL))                # closed form Q*(s, right) = gamma^(4-s)
print("Q(s, right) =", np.round(Q[:GOAL, 1], 4))  # -> [0.6559 0.729  0.81   0.9    1.    ]
print("optimal     =", np.round(opt, 4))          # -> [0.6561 0.729  0.81   0.9    1.    ]
greedy = np.argmax(Q, axis=1)[:GOAL]              # greedy action in each non-goal state
print("greedy path :", "".join("R" if a == 1 else "L" for a in greedy))  # -> RRRRR
