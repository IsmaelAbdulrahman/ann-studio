# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Loading datasets: one line each
# Code example 5 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

# 1) torchvision — vision sets, downloaded and cached on first use
from torchvision import datasets
mnist = datasets.MNIST("data", train=True,  download=True)
cifar = datasets.CIFAR10("data", train=True, download=True)   # also CIFAR100, SVHN, ...

# 2) Hugging Face 'datasets' — text, audio, tabular, images (and streamable)
from datasets import load_dataset
imdb = load_dataset("imdb")                       # train / test
sst2 = load_dataset("glue", "sst2")               # a named GLUE sub-task
c4   = load_dataset("c4", "en", streaming=True)   # too big for disk -> stream it

# 3) scikit-learn — the small classics, straight into memory
from sklearn.datasets import load_iris, fetch_openml
iris  = load_iris()                               # (150, 4) arrays
adult = fetch_openml("adult", version=2, as_frame=True)   # hundreds more via OpenML
