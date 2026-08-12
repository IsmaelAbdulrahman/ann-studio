# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 24: Uncertainty, calibration & Bayesian nets
# Section: Uncertainty in code
# Code example 3 of 5 in this chapter · PyTorch listing — copy into Google Colab or a local environment (not run in the app).
# Location: textbook / ANN Studio app, chapter "uncertainty"
# ====================================================================

import torch, torch.nn as nn, torch.nn.functional as F

# ---- MC Dropout: keep Dropout ON at test time, average T stochastic passes ----
@torch.no_grad()
def mc_dropout_predict(model, x, T=50):
    model.eval()                              # BatchNorm etc. in eval mode ...
    for m in model.modules():                 # ... but re-enable Dropout (Chapter 12)
        if isinstance(m, nn.Dropout):
            m.train()
    probs = torch.stack([F.softmax(model(x), dim=-1) for _ in range(T)])  # T x N x C
    mean = probs.mean(0)                       # predictive distribution
    var  = probs.var(0)                        # epistemic spread, per class
    entropy = -(mean * mean.clamp_min(1e-12).log()).sum(-1)               # total uncertainty
    return mean, var, entropy

# ---- Temperature scaling: fit ONE scalar T on a held-out validation set ----
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))     # parametrize T = exp(log_T) > 0
    def forward(self, logits):
        return logits / self.log_T.exp()

def fit_temperature(logits_val, labels_val):
    scaler = TemperatureScaler()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(scaler(logits_val), labels_val)   # NLL on validation logits
        loss.backward()
        return loss
    opt.step(closure)
    return scaler.log_T.exp().item()          # learned T, typically > 1 for over-confident nets
