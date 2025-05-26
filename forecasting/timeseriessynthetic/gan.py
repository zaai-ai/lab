import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, segment_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, segment_size)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, segment_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(segment_size, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def gan_train_eval(dataloader, t, segment_size, latent_dim):
    G = Generator(segment_size)
    D = Discriminator(segment_size)
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(G.parameters(), lr=0.001)
    d_optimizer = optim.Adam(D.parameters(), lr=0.001)

    for epoch in range(1000):
        d_loss_epoch = 0.0
        g_loss_epoch = 0.0
        num_batches = 0
        for real_batch in dataloader:
            real_data = real_batch[0]
            current_batch_size = real_data.size(0)  # may be < batch_size at end

            real_labels = torch.ones(current_batch_size, 1)
            fake_labels = torch.zeros(current_batch_size, 1)

            # discriminator training
            z = torch.randn(current_batch_size, latent_dim)
            fake_data = G(z)

            d_optimizer.zero_grad()
            d_real_loss = criterion(D(real_data), real_labels[: len(real_data)])
            d_fake_loss = criterion(
                D(fake_data.detach()), fake_labels[: len(real_data)]
            )
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # generator training
            g_optimizer.zero_grad()
            g_loss = criterion(D(fake_data), real_labels[: len(real_data)])
            g_loss.backward()
            g_optimizer.step()

            d_loss_epoch += d_loss.item()
            g_loss_epoch += g_loss.item()
            num_batches += 1

        if (epoch + 1) % 100 == 0 or epoch == 0:
            avg_d_loss = d_loss_epoch / num_batches
            avg_g_loss = g_loss_epoch / num_batches
            print(
                f"Epoch {epoch + 1:4d} | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
            )

    print("Training complete. Sampling from generator...")

    G.eval()
    with torch.no_grad():
        z = torch.randn(len(t) // segment_size + 1, latent_dim)
        gan_samples = G(z).numpy()
        gan_ts = np.concatenate(gan_samples)[: len(t)]

    return gan_ts
