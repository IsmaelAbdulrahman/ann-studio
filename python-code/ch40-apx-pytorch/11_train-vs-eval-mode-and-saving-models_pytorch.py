# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 40: Appendix F · PyTorch primer
# Section: Train vs eval mode, and saving models
# Code example 11 of 15 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "apx-pytorch"
# ====================================================================

import torch
import torch.nn as nn

model = nn.Linear(4, 3)

# ---- inference: eval mode + no gradient tracking ----
model.eval()
with torch.no_grad():
    preds = model(torch.randn(16, 4)).argmax(dim=1)
print("predicted classes:", preds[:5])

# ---- save and load the learned weights ----
torch.save(model.state_dict(), "model.pt")

fresh = nn.Linear(4, 3)                      # same architecture
fresh.load_state_dict(torch.load("model.pt"))
fresh.eval()
print("reloaded OK")
