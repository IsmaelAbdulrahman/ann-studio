# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 42: Appendix H · Datasets & further reading
# Section: Text datasets: IMDB and friends
# Code example 3 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-datasets"
# ====================================================================

# modern route: the Hugging Face 'datasets' + 'transformers' libraries
from datasets import load_dataset
from transformers import AutoTokenizer

imdb = load_dataset("imdb")                 # train / test splits of reviews
print(imdb["train"][0]["label"], imdb["train"][0]["text"][:60])

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
batch = tok(imdb["train"][:4]["text"], padding=True,
            truncation=True, return_tensors="pt")
print(batch["input_ids"].shape)             # (4, sequence_length)
