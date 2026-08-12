# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 27: Efficiency & deployment
# Section: Exercises
# Code example 4 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "efficiency"
# ====================================================================

import numpy as np
def reduction(k, Cin, Cout):
    standard = k*k*Cin*Cout
    dwsep    = k*k*Cin + Cin*Cout
    return standard, dwsep, standard/dwsep
for k in (3, 5):
    std, ds, r = reduction(k, 512, 512)
    print(f"k={k}: standard={std}  dw-sep={ds}  reduction={r:.2f}x")
# k=3: ~8.84x   k=5: ~23.84x
