# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 39: Appendix E · Python & NumPy primer
# Section: Lists, tuples and dictionaries
# Code example 2 of 22 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "apx-python"
# ====================================================================

# list: ordered and mutable
layers = [4, 8, 8, 1]
layers.append(1)                 # add to the end
print("list:", layers, "len", len(layers), "first", layers[0], "last", layers[-1])

# tuple: ordered and immutable (good for fixed shapes)
shape = (28, 28)
print("tuple:", shape, "pixels", shape[0] * shape[1])

# dict: key -> value lookup
cfg = {"lr": 0.01, "epochs": 20}
cfg["batch"] = 32                # add a new key
print("lr:", cfg["lr"], " momentum:", cfg.get("momentum", "default 0.9"))
