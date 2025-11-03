# vae_beam.py
import math
import os
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from functools import partial
import random

# -------------------------
# Utilities & abstractions
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ConvND(dim, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    if dim == 2:
        return nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    elif dim == 3:
        return nn.Conv3d(in_ch, out_ch, kernel_size, stride, padding)
    else:
        raise ValueError("dim must be 2 or 3")

def ConvTransposeND(dim, in_ch, out_ch, kernel_size=3, stride=1, padding=1, output_padding=0):
    if dim == 2:
        return nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
    elif dim == 3:
        return nn.ConvTranspose3d(in_ch, out_ch, kernel_size, stride, padding, output_padding)
    else:
        raise ValueError("dim must be 2 or 3")

def BatchNormND(dim, num_features):
    if dim == 2:
        return nn.BatchNorm2d(num_features)
    elif dim == 3:
        return nn.BatchNorm3d(num_features)

# -------------------------
# Encoder & Decoder blocks
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, dim, in_ch, out_ch, kernel=3, stride=1, padding=1, activation=nn.ReLU, batchnorm=True):
        super().__init__()
        self.conv = ConvND(dim, in_ch, out_ch, kernel, stride, padding)
        self.bn = BatchNormND(dim, out_ch) if batchnorm else None
        self.act = activation(inplace=True) if activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

# -------------------------
# VAE model
# -------------------------
class ConvVAE(nn.Module):
    def __init__(
        self,
        input_channels: int,
        latent_dim: int = 32,
        hidden_dims: Optional[list] = None,
        conv_dim: int = 2,  # 2 or 3
        input_spatial_shape: Tuple[int, ...] = (64, 64),
        use_batchnorm: bool = True,
    ):
        """
        input_channels: number of grid channels
        latent_dim: size of latent vector
        hidden_dims: list of feature depths for conv (will be used down the encoder)
        conv_dim: 2 or 3
        input_spatial_shape: input HxW or DxHxW (used to calculate flatten dims)
        """
        super().__init__()
        assert conv_dim in (2, 3)
        self.dim = conv_dim
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.use_batchnorm = use_batchnorm

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        # Encoder
        modules = []
        in_channels = input_channels
        for h in hidden_dims:
            modules.append(ConvBlock(self.dim, in_channels, h, kernel=4, stride=2, padding=1, batchnorm=use_batchnorm))
            in_channels = h
        self.encoder = nn.Sequential(*modules)

        # compute size after convs to build linear layers
        # pass a dummy tensor to infer
        with torch.no_grad():
            dummy_shape = (1, input_channels) + tuple(input_spatial_shape)
            dummy = torch.zeros(dummy_shape)
            enc_out = self.encoder(dummy)
            self.enc_shape = tuple(enc_out.shape[1:])  # (C, D/H, H, W) or (C,H,W)
            flat_dim = int(np.prod(enc_out.shape[1:]))
        self.flat_dim = flat_dim

        # Latent layers
        self.fc_mu = nn.Linear(flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(flat_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat_dim)

        # Decoder (mirror of encoder)
        decoder_modules = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_ch = hidden_dims_rev[0]
        # We'll create a sequence of ConvTranspose blocks
        for i in range(len(hidden_dims_rev) - 1):
            out_ch = hidden_dims_rev[i + 1]
            decoder_modules.append(
                nn.Sequential(
                    ConvTransposeND(self.dim, prev_ch, out_ch, kernel_size=4, stride=2, padding=1),
                    BatchNormND(self.dim, out_ch) if use_batchnorm else nn.Identity(),
                    nn.ReLU(inplace=True),
                )
            )
            prev_ch = out_ch

        # final transpose to reconstruct original channels
        # Note: output_padding used to ensure shape match; usually not needed if shapes match exactly
        decoder_modules = nn.ModuleList(decoder_modules)
        self.decoder_modules = decoder_modules
        # final conv to map to input channels
        self.final_layer = nn.Sequential(
            ConvND(self.dim, prev_ch, input_channels, kernel_size=3, stride=1, padding=1),
            nn.Softplus()  # ensures non-negative output (density). change to identity if not desired.
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        # reshape to conv shape
        shape = (z.size(0),) + tuple(self.enc_shape)
        h = h.view(shape)
        # pass through decoder modules
        for m in self.decoder_modules:
            h = m(h)
        x_rec = self.final_layer(h)
        return x_rec

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_rec = self.decode(z)
        return x_rec, mu, logvar, z

# -------------------------
# Physics-aware losses
# -------------------------
def poisson_nll_loss(pred, target, eps=1e-8):
    # target is count/density; pred is rate (>=0)
    # Poisson NLL per element: pred - target*log(pred) + log(target!) but the factorial term can be dropped for training
    return (pred - target * torch.log(pred + eps)).mean()

def total_charge_loss(pred, target):
    # Enforce conservation of total integrated density (L1 or relative)
    pred_charge = pred.sum(dim=tuple(range(1, pred.dim())))  # sum over channels/spatial dims -> (B,)
    target_charge = target.sum(dim=tuple(range(1, target.dim())))
    return F.l1_loss(pred_charge, target_charge)

def first_moment_loss(pred, target, coords):
    """
    coords: a tensor with shape (num_spatial_dims, D1, D2, ...) containing coordinate values along each axis
    pred, target shape: (B, C, D1, D2, ...)
    We'll compute center-of-mass per batch over all channels combined.
    """
    # combine channels by sum (density)
    pred_density = pred.sum(dim=1)
    target_density = target.sum(dim=1)
    # flatten spatial dims
    B = pred.shape[0]
    spatial_dims = pred_density.shape[1:]
    num_spatial = len(spatial_dims)
    pred_moments = []
    target_moments = []
    for axis in range(num_spatial):
        coord = coords[axis].to(pred.device)  # shape D1 x D2 x ...
        # expectation: sum(coord * density)/sum(density)
        pred_num = (pred_density * coord).view(B, -1).sum(dim=1)
        pred_den = pred_density.view(B, -1).sum(dim=1) + 1e-9
        target_num = (target_density * coord).view(B, -1).sum(dim=1)
        target_den = target_density.view(B, -1).sum(dim=1) + 1e-9
        pred_moments.append(pred_num / pred_den)
        target_moments.append(target_num / target_den)
    pred_m = torch.stack(pred_moments, dim=1)  # (B, num_spatial)
    target_m = torch.stack(target_moments, dim=1)
    return F.mse_loss(pred_m, target_m)

def covariance_loss(pred, target, coords):
    # Match covariance matrices (flattened) between pred and target
    # compute centered second moments
    pred_density = pred.sum(dim=1)
    target_density = target.sum(dim=1)
    B = pred.shape[0]
    num_spatial = len(pred_density.shape[1:])
    coords_flat = []
    for axis in range(num_spatial):
        coords_flat.append(coords[axis].to(pred.device).view(1, -1))  # (1, N)
    # compute mean positions
    pred_den = pred_density.view(B, -1).sum(dim=1, keepdim=True) + 1e-9
    target_den = target_density.view(B, -1).sum(dim=1, keepdim=True) + 1e-9

    pred_mean = []
    target_mean = []
    for axis in range(num_spatial):
        coord = coords_flat[axis]
        pnum = (pred_density.view(B, -1) * coord).sum(dim=1, keepdim=True)
        tnum = (target_density.view(B, -1) * coord).sum(dim=1, keepdim=True)
        pred_mean.append(pnum / pred_den)
        target_mean.append(tnum / target_den)
    pred_mean = torch.cat(pred_mean, dim=1)  # (B, num_spatial)
    target_mean = torch.cat(target_mean, dim=1)
    # compute covariance matrices
    losses = []
    for b in range(B):
        # construct Nx (num_spatial x N) coordinate matrix for sample b
        coord_stack = torch.stack([c.repeat(1, 1).view(-1) for c in coords_flat], dim=0).to(pred.device)  # (num_spatial, N)
        # centered coordinates
        cm = pred_mean[b].unsqueeze(1)  # (num_spatial, 1)
        centered = coord_stack - cm  # (num_spatial, N)
        w_pred = pred_density[b].view(-1)  # N
        w_tgt = target_density[b].view(-1)
        # weighted covariance: (centered @ (w * centered.T)) / sum(w)
        cov_pred = (centered * w_pred).matmul(centered.t()) / (w_pred.sum() + 1e-9)
        cov_tgt = (centered * w_tgt).matmul(centered.t()) / (w_tgt.sum() + 1e-9)
        losses.append(F.mse_loss(cov_pred, cov_tgt))
    return torch.stack(losses).mean()

# -------------------------
# Loss wrapper (ELBO + physics)
# -------------------------
def compute_loss(
    recon,
    x,
    mu,
    logvar,
    coords,
    recon_loss_type="mse",
    beta=1.0,
    physics_weights=None,
):
    """
    physics_weights: dict with keys like {'charge': w1, 'first_moment': w2, 'covariance': w3}
    """
    if recon_loss_type == "mse":
        rec_loss = F.mse_loss(recon, x)
    elif recon_loss_type == "poisson":
        rec_loss = poisson_nll_loss(recon, x)
    else:
        raise ValueError("Unknown recon_loss_type")

    # KL divergence term (closed form for Gaussian)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # physics losses
    physics_weights = {} if physics_weights is None else physics_weights
    p_loss = 0.0
    if physics_weights.get("charge", 0) > 0:
        p_loss = p_loss + physics_weights["charge"] * total_charge_loss(recon, x)
    if physics_weights.get("first_moment", 0) > 0:
        p_loss = p_loss + physics_weights["first_moment"] * first_moment_loss(recon, x, coords)
    if physics_weights.get("covariance", 0) > 0:
        p_loss = p_loss + physics_weights["covariance"] * covariance_loss(recon, x, coords)

    loss = rec_loss + beta * kld + p_loss
    return {
        "loss": loss,
        "reconstruction": rec_loss.detach(),
        "kld": kld.detach(),
        "physics": p_loss.detach() if isinstance(p_loss, torch.Tensor) else torch.tensor(0.0),
    }

# -------------------------
# Simple latent-space dynamics surrogate (example)
# -------------------------
class LatentDynamicsMLP(nn.Module):
    def __init__(self, latent_dim, hidden=[128, 128], dt=1.0):
        super().__init__()
        layers = []
        in_dim = latent_dim
        for h in hidden:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)
        self.dt = dt

    def forward(self, z):
        dz = self.net(z)
        # Euler step example: z_next = z + dt * dz
        return z + self.dt * dz

# -------------------------
# Dataset stub (you will replace with your loader)
# -------------------------
class GridParticleDataset(Dataset):
    """
    Placeholder dataset.
    Each sample: tensor of shape (C, D, H, W) or (C, H, W)
    """

    def __init__(self, numpy_arrays, coords=None, transform=None):
        # numpy_arrays: list or array of shape (N, C, *spatial)
        self.data = [torch.from_numpy(np.array(x, dtype=np.float32)) for x in numpy_arrays]
        self.transform = transform
        self.coords = coords

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x

# -------------------------
# Training loop
# -------------------------
def train_vae(
    model: ConvVAE,
    train_loader: DataLoader,
    coords: torch.Tensor,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    recon_loss_type: str = "mse",
    beta: float = 1.0,
    beta_schedule: Optional[dict] = None,
    physics_weights: Optional[dict] = None,
    save_every: int = 10,
    out_dir: str = "./checkpoints",
):
    os.makedirs(out_dir, exist_ok=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    model.to(device)
    # Beta schedule: dict with keys 'type': 'linear', 'start', 'end', 'epochs'
    def get_beta(e):
        if beta_schedule is None:
            return beta
        if beta_schedule.get("type") == "linear":
            s = beta_schedule.get("start", 0.0)
            e_end = beta_schedule.get("end", beta)
            total = beta_schedule.get("epochs", epochs)
            frac = min(1.0, e / float(total))
            return s + frac * (e_end - s)
        return beta

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        stats = {"loss": 0.0, "reconstruction": 0.0, "kld": 0.0, "physics": 0.0}
        current_beta = get_beta(epoch)
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, mu, logvar, z = model(batch)
            loss_dict = compute_loss(
                recon, batch, mu, logvar, coords=coords, recon_loss_type=recon_loss_type, beta=current_beta, physics_weights=physics_weights
            )
            loss_dict["loss"].backward()
            opt.step()
            running += 1
            for k in stats:
                stats[k] += float(loss_dict.get(k, 0.0))
        for k in stats:
            stats[k] /= max(1, running)
        print(f"Epoch {epoch}/{epochs} | loss {stats['loss']:.6f} | recon {stats['reconstruction']:.6f} | kld {stats['kld']:.6f} | phys {stats['physics']:.6f} | beta {current_beta:.4f}")
        if epoch % save_every == 0 or epoch == epochs:
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
            }, os.path.join(out_dir, f"vae_epoch_{epoch}.pt"))

# -------------------------
# Latent dynamics training (example)
# -------------------------
def train_latent_dynamics(
    vae: ConvVAE,
    dynamics_model: LatentDynamicsMLP,
    sequence_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    save_path: Optional[str] = None,
):
    """
    sequence_loader yields sequences: tensor shape (B, T, C, *spatial)
    We'll encode z_t and z_{t+1} and train an MSE on predicted z_{t+1}
    """
    vae.eval()
    dynamics_model.to(device)
    opt = torch.optim.Adam(dynamics_model.parameters(), lr=lr)
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        count = 0
        for seq in sequence_loader:
            seq = seq.to(device)
            B, T = seq.shape[0], seq.shape[1]
            # encode each time step (work in batch)
            with torch.no_grad():
                flat = seq.view(-1, *seq.shape[2:])  # (B*T, C, ...)
                mu, logvar = vae.encode(flat)
                z = vae.reparameterize(mu, logvar)
            z = z.view(B, T, -1)
            z_t = z[:, :-1, :].reshape(-1, z.shape[-1])
            z_tp1 = z[:, 1:, :].reshape(-1, z.shape[-1])
            z_pred = dynamics_model(z_t)
            loss = F.mse_loss(z_pred, z_tp1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            count += 1
        print(f"Latent-dyn epoch {epoch}/{epochs} | loss {running_loss/count:.6f}")
    if save_path:
        torch.save(dynamics_model.state_dict(), save_path)

# -------------------------
# Example usage (pseudo-code)
# -------------------------
if __name__ == "__main__":
    # reproducibility
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example: create synthetic dataset — replace with your data loader
    # We'll assume 2D grids 64x64 with 1 channel
    N = 256
    C = 1
    H = W = 64
    synthetic = []
    for i in range(N):
        # random gaussian blob(s) as placeholder
        grid = np.zeros((C, H, W), dtype=np.float32)
        num_blobs = np.random.randint(1, 4)
        for _ in range(num_blobs):
            cx = np.random.uniform(0.2 * W, 0.8 * W)
            cy = np.random.uniform(0.2 * H, 0.8 * H)
            sx = np.random.uniform(2, 8)
            sy = np.random.uniform(2, 8)
            xs = np.arange(W)[None, :]
            ys = np.arange(H)[:, None]
            blob = np.exp(-((xs - cx) ** 2 / (2 * sx ** 2) + (ys - cy) ** 2 / (2 * sy ** 2)))
            grid[0] += blob
        # small noise
        grid += 0.01 * np.random.randn(*grid.shape).astype(np.float32)
        grid = np.clip(grid, 0.0, None)
        synthetic.append(grid)
    ds = GridParticleDataset(synthetic)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)

    # Coordinates for physics losses (2D)
    xs = torch.tensor(np.linspace(0.0, 1.0, W), dtype=torch.float32)
    ys = torch.tensor(np.linspace(0.0, 1.0, H), dtype=torch.float32)
    # create grid coords shape (2, H, W)
    X, Y = torch.meshgrid(ys, xs, indexing='ij')
    coords = torch.stack([X, Y], dim=0)

    # Instantiate model
    vae = ConvVAE(input_channels=C, latent_dim=16, conv_dim=2, input_spatial_shape=(H, W))
    print("Model built. Params:", sum(p.numel() for p in vae.parameters() if p.requires_grad))

    # Train
    train_vae(
        model=vae,
        train_loader=loader,
        coords=coords,
        device=device,
        epochs=50,
        lr=1e-3,
        recon_loss_type="mse",
        beta_schedule={"type": "linear", "start": 0.0, "end": 1.0, "epochs": 20},
        physics_weights={"charge": 1.0, "first_moment": 1.0, "covariance": 0.1},
        save_every=10,
        out_dir="./vae_ckpts",
    )

    # After training, train latent dynamics (example)
    # Placeholder: build sequence dataset — you'll replace with real timesteps
    # sequence_loader yields (B, T, C, H, W)
    # For example purpose we create sequences by shifting blobs slightly
    seqs = []
    T = 5
    for i in range(200):
        base = synthetic[np.random.randint(len(synthetic))]
        seq = []
        for t in range(T):
            # small translation
            shift = t - T//2
            arr = np.roll(base, shift, axis=2).copy()
            seq.append(arr)
        seq = np.stack(seq, axis=0)  # (T, C, H, W)
        seqs.append(seq)
    seqs = np.stack(seqs, axis=0)  # (Nseq, T, C, H, W)
    seq_dataset = torch.tensor(seqs, dtype=torch.float32)
    seq_loader = DataLoader(seq_dataset, batch_size=8, shuffle=True)
    dyn = LatentDynamicsMLP(latent_dim=vae.latent_dim)
    train_latent_dynamics(vae, dyn, seq_loader, device=device, epochs=20, lr=1e-3, save_path="./latent_dyn.pt")
