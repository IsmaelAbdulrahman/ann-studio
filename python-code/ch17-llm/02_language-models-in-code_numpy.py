# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 17: Language models & the transformer era
# Section: Language models in code
# Code example 2 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "llm"
# ====================================================================

import numpy as np
np.random.seed(0)

# the model's probability for each ACTUAL next token in a 3-token target
p_true  = np.array([0.5, 0.2, 0.1])
ce      = -np.log(p_true)                         # per-token cross-entropy (nats)
mean_ce = ce.mean()
ppl     = np.exp(mean_ce)                         # perplexity = exp(mean CE)

print("per-token CE :", np.round(ce, 4))          # [0.6931 1.6094 2.3026]
print("mean CE      :", round(float(mean_ce), 4)) # 1.5351
print("perplexity   :", round(float(ppl), 4))     # 4.6416  = 10**(2/3)
