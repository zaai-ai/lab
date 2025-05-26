import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super(VAE, self).__init__()
        self.input_length = input_dim

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # [B, 16, L/2]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # [B, 32, L/4]
            nn.ReLU(),
        )

        conv_output_size = self.input_length // 4
        self.flatten_dim = 32 * conv_output_size

        # latent space
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_z = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=4, stride=2, padding=1),
            # nn.Tanh(),
        )

    def encode(self, x):
        x = x.unsqueeze(1)  # [B, 1, L]
        h = self.encoder(x)  # [B, 32, L/4]
        h_flat = h.view(h.size(0), -1)
        return self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_z(z)  # [B, flatten_dim]
        h = h.view(z.size(0), 32, self.input_length // 4)
        x_recon = self.decoder(h)  # [B, 1, L]
        return x_recon.squeeze(1)  # [B, L]

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div


def overlap_add_reconstruction(reconstructed_segments, segment_length=60, stride=1):
    num_segments, seg_len = reconstructed_segments.shape
    full_length = (num_segments - 1) * stride + seg_len

    recon = np.zeros(full_length)
    counts = np.zeros(full_length)

    for i, segment in enumerate(reconstructed_segments):
        start = i * stride
        end = start + segment_length
        recon[start:end] += segment
        counts[start:end] += 1

    # avoid divide-by-zero
    return recon / np.maximum(counts, 1e-8)


def vae_train_eval(
    dataloader,
    vae,
    segment_length,
    prior_ratio=0.3,  # x% of the time pick a pure prior sample
    beta_max=1.0,
    warmup_epochs=100,
    epochs=1000,
):
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        beta = beta_max * min(1.0, max(0.0, (epoch - warmup_epochs) / epochs))
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon, mu, logvar = vae(x)
            loss = vae_loss(recon, x, mu, logvar, beta)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            avg_loss = epoch_loss / num_batches
            print(f"Epoch {epoch + 1:4d} | VAE Loss: {avg_loss:.4f}")

    print("Training complete. Sampling from VAE...")

    vae.eval()
    with torch.no_grad():
        vae_samples_agg = []
        for batch in dataloader:
            x = batch[0]
            recon, mu, logvar = vae(x)
            z_post = vae.reparameterize(mu, logvar)
            z_prior = torch.randn_like(z_post)
            mask = torch.rand(z_post.size(0), 1) < prior_ratio
            z = torch.where(mask, z_prior, z_post)

            vae_samples = vae.decode(z).cpu().numpy()
            vae_samples_agg.append(vae_samples)
        vae_ts = np.concatenate(vae_samples_agg)
    vae_ts = overlap_add_reconstruction(vae_ts, segment_length=segment_length, stride=1)

    # vae.eval()
    # with torch.no_grad():
    #     for batch in dataloader:
    #         x = batch[0]
    #         recon, _, _ = vae(x)
    #         plt.plot(x[0].numpy(), label="Original")
    #         plt.plot(recon[0].numpy(), label="Reconstructed")
    #         plt.legend()
    #         plt.title("Reconstruction check")
    #         plt.show()

    return vae_ts
