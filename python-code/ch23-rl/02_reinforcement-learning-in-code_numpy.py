# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 23: Deep reinforcement learning
# Section: Reinforcement learning in code
# Code example 2 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "rl"
# ====================================================================

import numpy as np
np.random.seed(2)

# ---- 5-armed bandit with fixed, deliberately close true values ----
q_star = np.array([0.2, -0.3, 0.5, 0.0, 1.0])     # arm 4 is best (value 1.0)
k, best = len(q_star), int(np.argmax(q_star))

Q  = np.zeros(k); Nc = np.zeros(k)                 # value estimates and pull counts
eps, steps = 0.1, 2000
rewards = np.zeros(steps)
for t in range(steps):
    a = np.random.randint(k) if np.random.rand() < eps else int(np.argmax(Q))
    r = q_star[a] + np.random.randn()              # noisy reward, unit variance
    Nc[a] += 1
    Q[a]  += (r - Q[a]) / Nc[a]                     # incremental sample-average update
    rewards[t] = r

run = np.cumsum(rewards) / (np.arange(steps) + 1)  # running average reward
for c in [10, 50, 200, 500, 2000]:
    print("after %4d steps: avg reward %.3f" % (c, run[c - 1]))
#   -> 10:-0.007  50:0.482  200:0.752  500:0.791  2000:0.889  (rises toward 1.0)
print("random baseline mean(q*) = %.3f, best value = %.3f" % (q_star.mean(), q_star[best]))
print("final estimates Q =", np.round(Q, 3), " greedy arm =", int(np.argmax(Q)))  # arm 4
