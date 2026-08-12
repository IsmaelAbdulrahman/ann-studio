# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Loops and comprehensions
# Code example 4 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

# a plain loop
squares = []
for i in range(1, 6):
    squares.append(i * i)
print("loop         :", squares)

# the same result as a comprehension
print("comprehension:", [i * i for i in range(1, 6)])

# comprehension with a filter
print("evens 0..9   :", [n for n in range(10) if n % 2 == 0])

# enumerate and zip
for idx, nm in enumerate(["w1", "w2", "w3"]):
    print("   index", idx, "->", nm)
for a, b in zip([1, 2, 3], [10, 20, 30]):
    print("   pair", a, b)
