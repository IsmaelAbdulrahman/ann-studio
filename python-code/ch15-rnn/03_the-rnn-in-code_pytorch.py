# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 15: Recurrent networks & sequences
# Section: The RNN in code
# Code example 3 of 7 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "rnn"
# ====================================================================

import torch, torch.nn as nn

rnn = nn.RNN(input_size=1, hidden_size=8, batch_first=True)   # a tanh RNN cell
x = torch.randn(4, 6, 1)              # batch 4, sequence length 6, 1 feature/step
out, h_n = rnn(x)                     # out: every hidden state; h_n: the final one
print(out.shape, h_n.shape)          # torch.Size([4, 6, 8])  torch.Size([1, 4, 8])

# swap in an LSTM to defeat vanishing gradients — same call, plus a cell state:
lstm = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
out, (h_n, c_n) = lstm(x)
print(out.shape, c_n.shape)          # torch.Size([4, 6, 8])  torch.Size([1, 4, 8])
