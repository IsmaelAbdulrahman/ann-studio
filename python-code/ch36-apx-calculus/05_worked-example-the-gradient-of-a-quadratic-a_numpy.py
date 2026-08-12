# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 36: Appendix B · Calculus & the chain rule
# Section: A gradient dictionary → Worked example — the gradient of a quadratic at a point
# Code example 5 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-calculus"
# ====================================================================

import numpy as np
np.random.seed(0)
def grad_check(f, grad, x, h=1e-5):
    x = np.asarray(x, float)
    num = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e.flat[i] = h
        num.flat[i] = (f(x + e) - f(x - e)) / (2*h)   # central difference
    ana = grad(x)
    return np.max(np.abs(num - ana)) / (np.max(np.abs(ana)) + 1e-12)
# (1) quadratic  q(x) = 1/2 xᵀA x,  A symmetric  ->  grad = A x
A = np.array([[2.0, 1.0], [1.0, 3.0]])
q, q_grad = lambda x: 0.5 * x @ A @ x, lambda x: A @ x
x0 = np.array([1.0, -2.0])
print("quadratic  analytic", q_grad(x0), " rel.err %.1e" % grad_check(q, q_grad, x0))
# (2) log-sum-exp  ->  grad = softmax
def lse(z):     m = z.max(); return m + np.log(np.exp(z - m).sum())
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
z0 = np.array([2.0, 1.0, 0.0])
print("softmax    analytic", np.round(softmax(z0), 4), " rel.err %.1e" % grad_check(lse, softmax, z0))
print("both gradients confirmed to ~1e-10 relative error")
