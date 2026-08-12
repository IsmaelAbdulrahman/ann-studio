# ANN Studio — Python code from the textbook

Every Python code example from *Artificial Neural Networks* (Dr. Ismael Abdulrahman), organized by chapter.
Each file has a header comment naming the chapter, the section it appears in, and whether it is
**runnable NumPy** (also runs inside the ANN Studio app) or a **PyTorch listing** (copy into Colab).

**Totals:** 223 runnable NumPy examples · 50 PyTorch listings · 273 files across 43 chapters.

| # | Chapter | Folder | Code files |
|---|---|---|---|
| 1 | From biological to artificial neurons | `ch01-neuron/` | 6 |
| 2 | The perceptron & linear separability | `ch02-perceptron/` | 6 |
| 3 | Multilayer networks & the forward pass | `ch03-mlp/` | 7 |
| 4 | Activation functions | `ch04-activations/` | 9 |
| 5 | Features, embeddings & what a network learns | `ch05-representations/` | 5 |
| 6 | Loss functions & maximum likelihood | `ch06-loss/` | 8 |
| 7 | Gradient descent & its variants | `ch07-gd/` | 7 |
| 8 | Backpropagation | `ch08-backprop/` | 5 |
| 9 | Momentum, RMSProp & Adam | `ch09-optimizers/` | 6 |
| 10 | Initialization & the vanishing gradient | `ch10-init/` | 6 |
| 11 | Normalization: batch, layer & beyond | `ch11-normalization/` | 5 |
| 12 | Regularization & generalization | `ch12-regularization/` | 6 |
| 13 | Convolutional neural networks | `ch13-cnn/` | 6 |
| 14 | Modern CNN architectures & computer vision | `ch14-vision/` | 5 |
| 15 | Recurrent networks & sequences | `ch15-rnn/` | 7 |
| 16 | Attention & the transformer | `ch16-attention/` | 7 |
| 17 | Language models & the transformer era | `ch17-llm/` | 4 |
| 18 | Graph neural networks | `ch18-gnn/` | 5 |
| 19 | Autoencoders & representation learning | `ch19-autoencoders/` | 7 |
| 20 | Self-supervised & contrastive learning | `ch20-selfsup/` | 4 |
| 21 | Generative models: GANs to diffusion | `ch21-generative/` | 6 |
| 22 | Transfer learning & fine-tuning | `ch22-transfer/` | 5 |
| 23 | Deep reinforcement learning | `ch23-rl/` | 4 |
| 24 | Uncertainty, calibration & Bayesian nets | `ch24-uncertainty/` | 5 |
| 25 | Training in practice: the complete recipe | `ch25-practice/` | 8 |
| 26 | Data-centric deep learning | `ch26-data/` | 5 |
| 27 | Efficiency & deployment | `ch27-efficiency/` | 5 |
| 28 | Interpretability & explainability | `ch28-interpretability/` | 5 |
| 29 | Ethics, fairness & safety | `ch29-ethics/` | 6 |
| 30 | Case study: handwritten-digit recognition (MNIST) | `ch30-cs-mnist/` | 6 |
| 31 | Case study: tabular classification & churn | `ch31-cs-tabular/` | 7 |
| 32 | Case study: text sentiment analysis | `ch32-cs-text/` | 7 |
| 33 | Case study: time-series forecasting | `ch33-cs-sequence/` | 6 |
| 34 | Case study: anomaly detection with autoencoders | `ch34-cs-anomaly/` | 5 |
| 35 | Appendix A · Linear algebra refresher | `ch35-apx-linalg/` | 7 |
| 36 | Appendix B · Calculus & the chain rule | `ch36-apx-calculus/` | 6 |
| 37 | Appendix C · Probability & statistics | `ch37-apx-probability/` | 8 |
| 38 | Appendix D · Information theory | `ch38-apx-infotheory/` | 2 |
| 39 | Appendix E · Python & NumPy primer | `ch39-apx-python/` | 22 |
| 40 | Appendix F · PyTorch primer | `ch40-apx-pytorch/` | 15 |
| 41 | Appendix G · Autograd from scratch | `ch41-apx-autograd/` | 4 |
| 42 | Appendix H · Datasets & further reading | `ch42-apx-datasets/` | 7 |
| 43 | Appendix I · Formulas & symbols quick reference | `ch43-apx-formulas/` | 1 |

Run a NumPy example directly, e.g.:
```bash
python3 ch08-backprop/*_numpy.py
```

PyTorch listings need `pip install torch`; they are meant for Colab or a local setup.
