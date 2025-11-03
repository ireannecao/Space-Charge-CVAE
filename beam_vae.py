"""
beam_vae.py

Self-contained demo for:
- Latin Hypercube sampling of parameters
- Generating Gaussian-distributed 6D particle clouds from parameterized covariance matrices
- Making 2D histogram projections as image-like inputs (stacked channels)
- Training a convolutional VAE on these images
"""

import os
import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Optional visualization
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

# ---------------------------
# Utilities: Latin Hypercube
# ---------------------------
def latin_hypercube_sampling(n_samples: int, n_dims: int, rng: np.random.RandomState):
    """
    Simple Latin Hypercube Sampling (LHS).
    Returns array shape (n_samples, n_dims) with values in [0,1].
    """
    # create intervals
    cut = np.linspace(0, 1, n_samples + 1)
    u = rng.rand(n_samples, n_dims)
    a = cut[:n_samples]
    b = cut[1:n_samples + 1]
    points = a[:, None] + (b - a)[:, None] * u
    # permute each column
    H = np.zeros_like(points)
    for j in range(n_dims):
        order = rng.permutation(n_samples)
        H[:, j] = points[order, j]
    return H

# ---------------------------
# Covariance generation
# ---------------------------
def build_covariance_from_params(params: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Build a 6x6 positive-definite covariance matrix from a parameter vector.
    params: 1D array with length >= 6 (we'll use at least scales and optionally correlation factors).
    Strategy:
      - first 6 params -> log-scales for diagonal variances
      - remaining params define a low-rank factor or correlation mixing coefficients
    Returns Sigma (6x6)
    """
    d = 6
    # ensure params length
    if params.size < d:
        raise ValueError("params must have length >= 6")

    # diag scales: map from [0,1] to reasonable variance (e.g., 1e-6 .. 1e0) using logspace
    log_min, log_max = -6, 0
    scales = 10 ** (log_min + (log_max - log_min) * params[:d])
    D = np.diag(scales)

    # build a mixing matrix A of shape (6, r) where r = 3
    r = 3
    # use remaining params + rng noise to fill A
    leftovers = params[d:]
    needed = d * r
    vec = np.concatenate([
        leftovers,
        rng.normal(scale=0.1, size=max(0, needed - leftovers.size))
    ])[:needed]
    A = vec.reshape(d, r)
    # small scaling to avoid dominating diagonal scales
    A *= np.sqrt(np.mean(scales)) * 0.5

    # covariance = D + A A^T (positive definite)
    Sigma = D + A @ A.T

    # ensure symmetric positive definite by adding small jitter
    jitter = 1e-12 * np.eye(d)
    Sigma = (Sigma + Sigma.T) / 2.0 + jitter
    return Sigma

# ---------------------------
# Sample 6D Gaussian cloud
# ---------------------------
def sample_gaussian_cloud(Sigma: np.ndarray, n_particles: int, mean: np.ndarray = None, rng=None) -> np.ndarray:
    """
    Sample n_particles from 6D Gaussian with covariance Sigma
    Returns array shape (n_particles, 6)
    """
    d = Sigma.shape[0]
    if mean is None:
        mean = np.zeros(d)
    if rng is None:
        rng = np.random.RandomState()
    # Use cholesky for sampling
    L = np.linalg.cholesky(Sigma)
    z = rng.normal(size=(n_particles, d))
    samples = z @ L.T + mean[None, :]
    return samples

# ---------------------------
# Convert 6D cloud to stacked 2D hist images
# ---------------------------
def projections_from_cloud(cloud: np.ndarray,
                           projection_pairs: List[Tuple[int, int]],
                           image_size: int = 64,
                           range_padding: float = 0.05) -> np.ndarray:
    """
    Given cloud (N,6), compute for each (i,j) in projection_pairs a 2D histogram (image_size x image_size).
    Stack histograms into shape (C, H, W) where C=len(projection_pairs).
    """
    channels = []
    for (i, j) in projection_pairs:
        a = cloud[:, i]
        b = cloud[:, j]
        # compute symmetric range with small padding
        a_min, a_max = a.min(), a.max()
        b_min, b_max = b.min(), b.max()
        # if degenerate, expand a bit
        if a_max - a_min < 1e-8:
            a_min -= 1e-3; a_max += 1e-3
        if b_max - b_min < 1e-8:
            b_min -= 1e-3; b_max += 1e-3

        a_pad = (a_max - a_min) * range_padding
        b_pad = (b_max - b_min) * range_padding
        a_range = (a_min - a_pad, a_max + a_pad)
        b_range = (b_min - b_pad, b_max + b_pad)

        H, xedges, yedges = np.histogram2d(a, b, bins=image_size, range=[a_range, b_range], density=True)
        # histogram2d returns H with shape (nbins_x, nbins_y) — we want (H, W)
        # normalize and convert to float32
        H = H.astype(np.float32)
        channels.append(H)
    # stack channels -> (C,H,W)
    img = np.stack(channels, axis=0)
    return img

# ---------------------------
# Dataset
# ---------------------------
class BeamDataset(Dataset):
    def __init__(self,
                 n_examples: int = 2000,
                 n_particles: int = 20000,
                 projection_pairs: List[Tuple[int, int]] = None,
                 image_size: int = 64,
                 rng_seed: int = 42):
        self.n_examples = n_examples
        self.n_particles = n_particles
        self.image_size = image_size
        self.rng = np.random.RandomState(rng_seed)
        if projection_pairs is None:
            # typical physics pairs (x,px), (y,py), (z,pz)
            self.projection_pairs = [(0, 3), (1, 4), (2, 5)]
        else:
            self.projection_pairs = projection_pairs

        # Pre-sample LHS parameters
        n_params = 6 + 9  # allow extra param space
        lhs = latin_hypercube_sampling(self.n_examples, n_params, self.rng)
        self.params = lhs  # shape (n_examples, n_params)

        # optionally precompute images (memory cost). We'll compute-on-the-fly to keep memory low
        # but compute and store covariance and means per example for reproducibility
        self.covariances = []
        for k in range(self.n_examples):
            Sigma = build_covariance_from_params(self.params[k], self.rng)
            self.covariances.append(Sigma)

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):
        Sigma = self.covariances[idx]
        cloud = sample_gaussian_cloud(Sigma, self.n_particles, rng=self.rng)
        img = projections_from_cloud(cloud, self.projection_pairs, image_size=self.image_size)
        # Optionally apply small log transform to stabilize dynamic range
        img = np.log1p(img * 1e3)  # scale factor: tune as needed
        # Normalize each channel to zero-mean/unit-variance per sample (helps training)
        img = (img - img.mean()) / (img.std() + 1e-8)
        # Convert to float32 tensor
        img_t = torch.from_numpy(img).float()
        return img_t

# ---------------------------
# VAE model (conv encoder/decoder)
# ---------------------------
class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        # small conv stack; adjust channels if necessary
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),  # 64 -> 32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),  # 32 -> 16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 16 -> 8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 8 -> 4
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.flatten = nn.Flatten()
        # compute final feature size: assuming input 64x64 -> after 4 downsamples -> 4x4 with 256 channels
        self.feature_dim = 256 * 4 * 4
        self.fc_mu = nn.Linear(self.feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.feature_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
        # transpose convs mirroring encoder
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 4 -> 8
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 8 -> 16
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 16 -> 32
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, out_channels, 4, 2, 1),  # 32 -> 64
            # final activation: none; we'll use L1/MSE on output
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        x = self.deconv(x)
        return x

class ConvVAE(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, latent_dim)
        self.decoder = ConvDecoder(in_channels, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ---------------------------
# Loss function
# ---------------------------
def vae_loss(recon_x, x, mu, logvar, beta=1.0, recon_loss_type="mse"):
    """
    recon_x, x: tensors (B, C, H, W)
    mu, logvar: (B, latent)
    beta: weight on KL (for beta-VAE experiments)
    recon_loss_type: "mse" or "l1"
    """
    if recon_loss_type == "mse":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    elif recon_loss_type == "l1":
        recon_loss = F.l1_loss(recon_x, x, reduction="sum")
    else:
        raise ValueError("unknown recon_loss_type")

    # KL divergence between N(mu, var) and N(0,1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

# ---------------------------
# Training Loop
# ---------------------------
def train_vae(model: nn.Module,
              train_loader: DataLoader,
              val_loader: DataLoader = None,
              epochs: int = 50,
              lr: float = 1e-3,
              device: str = "cuda" if torch.cuda.is_available() else "cpu",
              beta: float = 1.0,
              recon_loss_type: str = "mse",
              save_path: str = None):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        t0 = time.time()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss, recon_l, kl_l = vae_loss(recon, batch, mu, logvar, beta=beta, recon_loss_type=recon_loss_type)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_recon += recon_l.item()
            train_kl += kl_l.item()

        scheduler.step()
        t1 = time.time()

        # Validation step (optional)
        val_loss = None
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                v_loss = 0.0
                v_recon = 0.0
                v_kl = 0.0
                for vb in val_loader:
                    vb = vb.to(device)
                    recon, mu, logvar = model(vb)
                    loss, recon_l, kl_l = vae_loss(recon, vb, mu, logvar, beta=beta, recon_loss_type=recon_loss_type)
                    v_loss += loss.item()
                    v_recon += recon_l.item()
                    v_kl += kl_l.item()
                val_loss = v_loss

        # Logging
        n_train = len(train_loader.dataset)
        print(f"Epoch {epoch:3d} | train_loss {train_loss / n_train:.6f} | train_recon {train_recon / n_train:.6f} | train_kl {train_kl / n_train:.6f} | time {t1 - t0:.1f}s", end="")
        if val_loss is not None:
            print(f" | val_loss {val_loss / len(val_loader.dataset):.6f}")
        else:
            print()

        # Save checkpoint occasionally
        if save_path is not None and (epoch % 10 == 0 or epoch == epochs):
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }, os.path.join(save_path, f"vae_epoch_{epoch}.pt"))

    return model

# ---------------------------
# Utilities: sample & latent operator
# ---------------------------
def sample_from_prior(model: ConvVAE, n: int = 16, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, model.encoder.fc_mu.out_features).to(device)
        x = model.decoder(z)
        return x.cpu()

def latent_linear_operator(z: torch.Tensor, A: torch.Tensor = None, b: torch.Tensor = None):
    """
    Applies a linear operator in latent space: z' = A z + b
    If A is None, returns scaled z (toy example).
    """
    if A is None:
        return 1.0 * z
    else:
        return z @ A.T + (b if b is not None else 0.0)

# ---------------------------
# Example usage (main)
# ---------------------------
def example_main():
    # Settings
    n_examples = 2000
    n_particles = 20000
    image_size = 64
    batch_size = 32
    latent_dim = 16
    epochs = 40
    projection_pairs = [(0, 3), (1, 4), (2, 5)]  # three channels

    # make dataset
    dataset = BeamDataset(n_examples=n_examples,
                          n_particles=n_particles,
                          projection_pairs=projection_pairs,
                          image_size=image_size,
                          rng_seed=1234)
    # split
    n_train = int(0.9 * len(dataset))
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # model
    in_channels = len(projection_pairs)
    model = ConvVAE(in_channels=in_channels, latent_dim=latent_dim)

    # train
    os.makedirs("checkpoints", exist_ok=True)
    trained = train_vae(model, train_loader, val_loader,
                        epochs=epochs, lr=1e-3, beta=1.0, recon_loss_type="mse",
                        save_path="checkpoints")

    # sample from prior and show
    samples = sample_from_prior(trained, n=4)
    # shape (n, C, H, W)
    if HAS_MPL:
        for i in range(samples.shape[0]):
            s = samples[i].numpy()
            # show first channel for quick look
            plt.figure(figsize=(4, 4))
            plt.title(f"Sample {i}")
            plt.imshow(s[0], origin="lower")
            plt.colorbar()
            plt.show()

    # Example latent-space op: take a validation batch, encode, apply linear operator, decode
    model.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader)).to(next(model.parameters()).device)
        mu, logvar = model.encoder(val_batch)
        z = model.reparameterize(mu, logvar)
        # toy operator: small rotation in latent subspace (2-d rotation on first two dims)
        A = torch.eye(z.size(-1))
        theta = 0.3
        c, s = math.cos(theta), math.sin(theta)
        if z.size(-1) >= 2:
            A[0, 0] = c; A[0, 1] = -s
            A[1, 0] = s; A[1, 1] = c
        z2 = latent_linear_operator(z, A=A.to(z.device))
        recon2 = model.decoder(z2)
        if HAS_MPL:
            # show original and transformed for first example
            orig = val_batch[0].cpu().numpy()
            new = recon2[0].cpu().numpy()
            plt.figure(figsize=(8, 4))
            for ci in range(orig.shape[0]):
                plt.subplot(2, orig.shape[0], ci + 1)
                plt.imshow(orig[ci], origin="lower"); plt.title(f"orig ch{ci}")
                plt.axis("off")
                plt.subplot(2, orig.shape[0], orig.shape[0] + ci + 1)
                plt.imshow(new[ci], origin="lower"); plt.title(f"latent-op ch{ci}")
                plt.axis("off")
            plt.show()

if __name__ == "__main__":
    example_main()
