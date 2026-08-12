# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Variables, numbers and text
# Code example 1 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

# Python infers the type from the value (dynamic typing)
count = 3               # int
rate  = 0.05           # float
name  = "layer"        # str
ok    = True           # bool

print(type(count).__name__, type(rate).__name__, type(name).__name__)
print("floor division 7 // 2 =", 7 // 2)   # -> 3
print("true  division 7 /  2 =", 7 / 2)    # -> 3.5
print("power        2 ** 10 =", 2 ** 10)   # -> 1024
print(name + "_" + str(count))             # join text with str()
