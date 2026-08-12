# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 41: Appendix G · Autograd from scratch
# Section: Exercises
# Code example 4 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-autograd"
# ====================================================================

import numpy as np
class Value:
    def __init__(self, data, _children=()):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)
    def __add__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data + o.data, (self, o))
        def _b(): self.grad += out.grad; o.grad += out.grad
        out._backward = _b; return out
    def __mul__(self, o):
        o = o if isinstance(o, Value) else Value(o)
        out = Value(self.data * o.data, (self, o))
        def _b(): self.grad += o.data * out.grad; o.grad += self.data * out.grad
        out._backward = _b; return out
    def __pow__(self, k):
        out = Value(self.data ** k, (self,))
        def _b(): self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _b; return out
    def tanh(self):
        t = np.tanh(self.data); out = Value(t, (self,))
        def _b(): self.grad += (1 - t * t) * out.grad
        out._backward = _b; return out
    def __neg__(self):    return self * -1
    def __sub__(self, o): return self + (-o)
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev: build(p)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()

w, x, b, y = Value(1.0), Value(0.5), Value(0.5), Value(1.0)
def loss():
    return ((w * x + b).tanh() - y) ** 2
loss().backward()                          # fills w.grad, b.grad, ...
print(f"analytic dL/db = {b.grad:.6f}")    # -> -0.200249  (the figure's -0.200)

eps = 1e-6; o = b.data                      # central-difference check on the bias
b.data = o + eps; Lp = loss().data
b.data = o - eps; Lm = loss().data
b.data = o
print(f"numeric  dL/db = {(Lp - Lm) / (2 * eps):.6f}")
