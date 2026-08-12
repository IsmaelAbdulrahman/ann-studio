# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Tabular datasets: Iris and the UCI repository
# Code example 2 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

import pandas as pd
import torch

df = pd.read_csv("churn.csv")
y = torch.tensor(df.pop("churn").values, dtype=torch.long)     # target column
X = torch.tensor(df.values, dtype=torch.float32)               # the features
X = (X - X.mean(0)) / (X.std(0) + 1e-8)                        # standardize
print(X.shape, y.shape)

# scikit-learn ships the small classics directly:
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.data.shape, iris.target[:5])   # (150, 4)  [0 0 0 0 0]
