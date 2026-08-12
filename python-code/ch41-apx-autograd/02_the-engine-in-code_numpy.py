# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 41: Appendix G · Autograd from scratch
# Section: The engine, in code
# Code example 2 of 4 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-autograd"
# ====================================================================

import numpy as np
np.random.seed(1)

class Value:
    """A scalar node in a dynamic autodiff graph."""
    def __init__(self, data, _children=()):
        self.data = float(data)          # the forward value
        self.grad = 0.0                  # dL/d(self), filled by backward()
        self._backward = lambda: None    # local rule: push grad to parents
        self._prev = set(_children)      # the Values this one was built from

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad  += out.grad       # d(a+b)/da = 1
            other.grad += out.grad       # fan-out: ACCUMULATE, never overwrite
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad  += other.data * out.grad     # d(ab)/da = b
            other.grad += self.data  * out.grad     # d(ab)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):                           # k is a plain number
        out = Value(self.data ** k, (self,))
        def _backward():
            self.grad += k * self.data ** (k - 1) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Value(t, (self,))
        def _backward():
            self.grad += (1 - t * t) * out.grad     # tanh'(z) = 1 - tanh(z)^2
        out._backward = _backward
        return out

    def __neg__(self):         return self * -1
    def __sub__(self, other):  return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    def backward(self):
        topo, seen = [], set()
        def build(v):                     # depth-first topological sort
            if v not in seen:
                seen.add(v)
                for parent in v._prev:
                    build(parent)
                topo.append(v)            # node appended AFTER its parents
        build(self)
        self.grad = 1.0                   # seed dL/dL = 1
        for v in reversed(topo):          # sweep in reverse topological order
            v._backward()

# ---- a tiny 2-input -> 2-hidden(tanh) -> 1-output network of Values ----
rv = lambda: Value(np.random.randn() * 0.5)    # one seeded random parameter
W1 = [[rv(), rv()], [rv(), rv()]]    # W1[j][i]: hidden unit j, input i
b1 = [rv(), rv()]
W2 = [rv(), rv()]                     # output weights
b2 = rv()
params = [w for row in W1 for w in row] + b1 + W2 + [b2]

x = [Value(0.5), Value(-1.0)]         # one input example
y = Value(1.0)                        # its target

def forward(x):
    h = []
    for j in range(2):                # hidden layer, tanh units
        s = b1[j]
        for i in range(2):
            s = s + W1[j][i] * x[i]
        h.append(s.tanh())
    o = b2                            # linear output unit
    for j in range(2):
        o = o + W2[j] * h[j]
    return o

pred = forward(x)
loss = (pred - y) ** 2                # squared error on one example
loss.backward()                      # ONE reverse sweep fills every .grad
print(f"pred = {pred.data:.4f}   loss = {loss.data:.4f}")
print(f"dL/dW2[0] = {W2[0].grad:.6f}   dL/db2 = {b2.grad:.6f}")

# ---- finite-difference check on one parameter (central difference) ----
def loss_value():                    # rebuild the graph at the current params
    return ((forward(x) - y) ** 2).data
p = W2[0]; g = p.grad; o = p.data; eps = 1e-6
p.data = o + eps; Lp = loss_value()
p.data = o - eps; Lm = loss_value()
p.data = o                                              # restore
print(f"grad check W2[0]: analytic {g:.6f}   numeric {(Lp - Lm) / (2 * eps):.6f}")

# ---- gradient-descent loop: the engine can now TRAIN the network ----
for step in range(200):
    loss = (forward(x) - y) ** 2
    for q in params: q.grad = 0.0    # zero_grad -- grads accumulate, so reset!
    loss.backward()                  # backprop
    for q in params: q.data -= 0.1 * q.grad    # SGD step:  theta <- theta - eta*grad
print(f"after 200 steps: loss = {loss.data:.2e}   pred = {forward(x).data:.4f}  (target 1.0)")
