import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from cvae import cVAE
from beam_data import generate_frequency_maps
import os

# ============================================================
# === Torch Dataset for Gaussian Maps ===
# ============================================================
class FrequencyMapDataset(Dataset):
    def __init__(self, filename="frequency_maps.npz"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"{filename} not found. Please generate it first.")
        npzfile = np.load(filename)
        self.data = npzfile["data"]  # shape (N, 15, bins, bins)
        self.data = torch.from_numpy(self.data).float()  # convert to tensor

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx]  # shape (15, bins, bins)


# ============================================================
# === Instantiate and Train the cVAE ===
# ============================================================
def train_cvae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 16
    epochs = 20
    learning_rate = 1e-3

    # --------------------------
    # Dataset and DataLoader
    # --------------------------
    dataset = FrequencyMapDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # --------------------------
    # Instantiate cVAE
    # --------------------------
    model = cVAE(
        imch=15,        # 15 planes
        f1=16, f2=32, f3=64, f4=128, f5=256,
        n1=512, n2=0, n3=256,
        d1=1, imfinal1=2, imfinal2=2,  # imfinal1/2 can be dummy if dynamic
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()

    # --------------------------
    # Training loop
    # --------------------------
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for x in loop:
            x = x.to(device)
            optimizer.zero_grad()
            x_hat, mean, logvar, z = model(x)
            
            # VAE loss: reconstruction + KL divergence
            recon_loss = mse_loss(x_hat, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean**2 - torch.exp(logvar))
            loss = recon_loss + kl_loss
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x.size(0)
            loop.set_postfix(loss=loss.item())


        epoch_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.6f}")

    # --------------------------
    # Save model
    # --------------------------
    torch.save(model.state_dict(), "cvae_gaussian_maps3.pth")
    print("Model saved to cvae_gaussian_maps3.pth")



if __name__ == "__main__":
    train_cvae()
