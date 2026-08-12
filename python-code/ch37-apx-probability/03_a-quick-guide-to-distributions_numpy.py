# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 37: Appendix C · Probability & statistics
# Section: A quick guide to distributions
# Code example 3 of 8 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-probability"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)
n = 200000
p, lam, beta = 0.3, 4.0, 2.0                  # Bernoulli p, Poisson rate, Exp scale
tests = [
    ("Bernoulli(0.3)",   rng.binomial(1, p, n),    p,      p*(1-p)),
    ("Binomial(10,0.3)", rng.binomial(10, p, n),    10*p,   10*p*(1-p)),
    ("Poisson(4)",       rng.poisson(lam, n),       lam,    lam),
    ("Exponential(2)",   rng.exponential(beta, n),  beta,   beta**2),
    ("Uniform(0,1)",     rng.uniform(0, 1, n),      0.5,    1/12),
]
for name, x, m, v in tests:                    # empirical moments match theory
    print(f"{name:16s} mean {x.mean():6.3f} (th {m:5.3f})   var {x.var():7.3f} (th {v:6.3f})")
