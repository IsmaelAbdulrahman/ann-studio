# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Functions
# Code example 3 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

def affine(x, w=1.0, b=0.0):
    """A tiny linear unit w*x + b (the neuron of chapter 1). w, b default to 1, 0."""
    return w * x + b

print(affine(3))                  # defaults -> 1*3 + 0 = 3.0
print(affine(3, w=2.0))           # keyword arg -> 2*3 + 0 = 6.0
print(affine(3, w=2.0, b=1.0))    # -> 2*3 + 1 = 7.0

square = lambda x: x * x          # a one-line anonymous function
print("square(5) =", square(5))
