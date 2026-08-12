# ====================================================================
# Artificial Neural Networks — Dr. Ismael Khorshed Abdulrahman, PhD (Electrical & Computer Engineering), Tennessee Technological University, USA
# Chapter 21: Generative models: GANs to diffusion
# Section: Generative models in code
# Code example 2 of 6 in this chapter · Runnable NumPy — runs inside the ANN Studio app, and standalone with NumPy.
# Location: textbook / ANN Studio app, chapter "generative"
# ====================================================================

import numpy as np
rng = np.random.RandomState(0)

def gauss(x, mu, s=1.0):
    return np.exp(-(x - mu) ** 2 / (2 * s * s)) / np.sqrt(2 * np.pi * s * s)

# real data ~ N(3,1). Generator emits N(gmu,1); slide gmu toward 3.
def game_value(gmu, n=200000):
    real = rng.normal(3, 1, n)
    fake = rng.normal(gmu, 1, n)
    # best-response discriminator for THIS generator: D*=p_data/(p_data+p_G)
    Dr = gauss(real, 3) / (gauss(real, 3) + gauss(real, gmu))
    Df = gauss(fake, 3) / (gauss(fake, 3) + gauss(fake, gmu))
    return np.log(Dr).mean() + np.log(1 - Df).mean()

for gmu in [0.0, 1.0, 2.0, 3.0]:
    print(f"generator N({gmu:.0f},1):  max_D V = {game_value(gmu):+.3f}")
print(f"floor  2*ln(0.5)      = {2 * np.log(0.5):+.3f}")
