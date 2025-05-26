import numpy as np
from tslearn.barycenters import dtw_barycenter_averaging
import torch
from torch.utils.data import DataLoader, TensorDataset

from forecasting.timeseriessynthetic.transformations import (
    parametric_transformations,
    block_bootstrap,
    ts_mixup,
    gmm_sampling,
    stl_mbb,
)
from forecasting.timeseriessynthetic.vae import vae_train_eval, VAE
from forecasting.timeseriessynthetic.gan import gan_train_eval
from forecasting.timeseriessynthetic.visualizations import (
    plot_augmentations,
    plot_advanced_mix,
    plot_generative_models,
)

np.random.seed(42)


def create_base_series(t: np.ndarray) -> np.ndarray:
    return np.sin(t) + 0.1 * np.random.randn(len(t))


def main():
    t = np.linspace(0, 10, 500)
    base_series = create_base_series(t)

    # classic augmentations
    jittered, scaled = parametric_transformations(base_series)
    bootstrap_series = block_bootstrap(base_series)

    # advanced mix
    mixup_series = ts_mixup(jittered, scaled)
    dba_series = dtw_barycenter_averaging(np.vstack([base_series, jittered, scaled]))
    stl_mbb_series = stl_mbb(base_series)

    # probabilistic modeling
    segment_size = 40
    prob_model_series, segments = gmm_sampling(base_series, segment_size=segment_size)

    # dataLoader for deep models
    X_tensor = torch.tensor(segments)
    dataloader = DataLoader(TensorDataset(X_tensor), batch_size=16, shuffle=False)

    # generative models
    latent_dim = 50
    vae = VAE(latent_dim=latent_dim, input_dim=segment_size)

    vae_ts = vae_train_eval(dataloader, vae=vae, segment_length=segment_size)
    gan_ts = gan_train_eval(dataloader, t, segment_size=segment_size, latent_dim=10)

    # visualizations
    plot_augmentations(t, base_series, jittered, scaled, bootstrap_series)
    plot_advanced_mix(t, base_series, mixup_series, dba_series, stl_mbb_series)
    plot_generative_models(t, base_series, prob_model_series, vae_ts, gan_ts)


if __name__ == "__main__":
    main()
