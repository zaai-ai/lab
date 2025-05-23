import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.mixture import GaussianMixture
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from forecasting.timeseriessynthetic.vae import vae_train_eval
from forecasting.timeseriessynthetic.gan import gan_train_eval

np.random.seed(42)

# create a base time series (sinusoidal signal with noise)
t = np.linspace(0, 10, 500)
base_series = np.sin(t) + 0.1 * np.random.randn(len(t))

# parametric transformations
jittered = base_series + 0.2 * np.random.randn(len(t))
scaled = 1.5 * base_series

# pattern-mixing / block bootstrap
block_size = 50
blocks = [base_series[i:i + block_size] for i in range(0, len(base_series) - block_size, block_size)]
random.shuffle(blocks)
bootstrap_series = np.concatenate(blocks[:len(t) // block_size])

# probabilistic model: fit GMM to segments and sample new ones
segment_size = 100
segments = np.array([base_series[i:i+segment_size] for i in range(0, len(base_series)-segment_size)])
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(segments)
samples, _ = gmm.sample(len(t) // segment_size)
prob_model_series = np.concatenate(samples)

# ----
segments = segments.astype(np.float32)

X_tensor = torch.tensor(segments)
dataset = TensorDataset(X_tensor)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# vae
vae_ts = vae_train_eval(dataloader, t, segment_size)

# gan
gan_ts = gan_train_eval(dataloader, t, segment_size, batch_size)


fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t, base_series, label='Original Series', color='black')
axs[0].set_title('Original for Reference')
axs[0].legend()

axs[1].plot(t, base_series, label='Original Series')
axs[1].plot(t, jittered, label='Jittered', alpha=0.7)
axs[1].plot(t, scaled, label='Scaled', alpha=0.7)
axs[1].set_title('Parametric Transformations')
axs[1].legend()

axs[2].plot(t[:len(bootstrap_series)], bootstrap_series, label='Block Bootstrap', color='orange')
axs[2].set_title('Pattern-Mixing / Block Bootstrap')
axs[2].legend()

plt.tight_layout()
plt.show()


fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(t, base_series, label='Original Series', color='black')
axs[0].set_title('Original for Reference')
axs[0].legend()

axs[1].plot(t[:len(prob_model_series)], prob_model_series, label='Sampled from GMM', color='green')
axs[1].set_title('Probabilistic Generative Model (GMM)')
axs[1].legend()

axs[2].plot(t, vae_ts, label='Sampled from VAE', color='blue')
axs[2].set_title('VAE')
axs[2].legend()

axs[3].plot(t, gan_ts, label='Sampled from GAN', color='red')
axs[3].set_title('GAN')
axs[3].legend()

plt.tight_layout()
plt.show()