# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 32: Case study: text sentiment analysis
# Section: In code
# Code example 4 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "cs-text"
# ====================================================================

import torch, torch.nn as nn, torch.nn.functional as F

# Real data: the IMDB dataset, e.g.
#   from datasets import load_dataset
#   imdb = load_dataset("imdb")               # 25k train / 25k test
# Build a vocab (id 0 = <pad>, id 1 = <unk>) from training tokens, map each
# review to a list of ids, then pad each batch to a rectangle (ids, mask).

class SentimentNet(nn.Module):
    def __init__(self, vocab_size, dim=64, hidden=64, pad_idx=0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=pad_idx)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)                  # a single logit
    def forward(self, ids, mask):                        # ids,mask: (B, L)
        e = self.emb(ids)                                # (B, L, dim)
        summed = (e * mask.unsqueeze(-1)).sum(1)         # ignore padding
        avg = summed / mask.sum(1, keepdim=True).clamp(min=1)   # mean pool
        return self.fc2(F.relu(self.fc1(avg))).squeeze(1)

net = SentimentNet(vocab_size=20000)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(5):
    net.train()
    for ids, mask, y in train_loader:                    # y: float in {0., 1.}
        opt.zero_grad()
        loss_fn(net(ids, mask), y).backward()
        opt.step()
    print(f"epoch {epoch} done")

# Word order matters ("good, not bad" vs "bad, not good"): swap mean pooling
# for a recurrent encoder ->  nn.LSTM(dim, hidden, batch_first=True), take the
# final hidden state (chapter 15); or a Transformer encoder for self-attention (chapter 16).
