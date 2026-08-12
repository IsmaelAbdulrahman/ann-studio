# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 22: Transfer learning & fine-tuning
# Section: Transfer learning in code
# Code example 2 of 5 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "transfer"
# ====================================================================

import numpy as np
np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)

def softmax_T(logits, T):                    # temperature-scaled softmax
    z = np.asarray(logits) / T
    z = z - z.max()                          # stabilize before exp
    e = np.exp(z)
    return e / e.sum()

teacher = np.array([3.0, 1.0, 0.5])          # teacher logits over 3 classes
student = np.array([2.0, 1.0, 1.0])          # student logits (being trained)

pt1 = softmax_T(teacher, 1.0)                # teacher at T = 1 (sharp)
pt2 = softmax_T(teacher, 2.0)                # teacher at T = 2 (softened)
print("teacher T=1:", pt1, " max =", round(float(pt1.max()), 4))   # 0.8214
print("teacher T=2:", pt2, " max =", round(float(pt2.max()), 4))   # 0.6045  -> T>1 softens

# KL( teacher_soft || student_soft ) at T = 2  (the distillation objective)
T = 2.0
p = softmax_T(teacher, T)
q = softmax_T(student, T)
kl = np.sum(p * np.log(p / q))
print("student T=2:", q)
print("KL(teacher||student) at T=2 =", round(float(kl), 4), "nats")  # 0.0499

# raising T raises the entropy of the target -> a softer, more informative signal
H1 = -np.sum(pt1 * np.log(pt1)); H2 = -np.sum(pt2 * np.log(pt2))
print("entropy T=1 =", round(float(H1), 4), " entropy T=2 =", round(float(H2), 4))  # 0.5876 -> 0.9423
