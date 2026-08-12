# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 41: Appendix G · Autograd from scratch
# Section: Exercises
# Code example 3 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-autograd"
# ====================================================================

import numpy as np
class Value:
    def __init__(self, data, _children=()):
        self.data = float(data); self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)
    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,))
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad   # 1 if z>0 else 0
        out._backward = _backward
        return out
    def backward(self):
        topo, seen = [], set()
        def build(v):
            if v not in seen:
                seen.add(v)
                for p in v._prev: build(p)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()

z = Value(-2.0); a = z.relu(); a.backward()
print("relu(-2) =", a.data, "  d/dz =", z.grad)   # 0.0   0.0
z = Value( 3.0); a = z.relu(); a.backward()
print("relu( 3) =", a.data, "  d/dz =", z.grad)   # 3.0   1.0
