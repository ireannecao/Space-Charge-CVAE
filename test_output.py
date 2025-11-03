import torch
import matplotlib.pyplot as plt
import numpy as np
from beam_data import generate_frequency_maps, PLANE_MAP
from cvae import cVAE

# --------------------------
# Load trained model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = cVAE(
    imch=15,        # number of planes
    f1=16, f2=32, f3=64, f4=128, f5=256,
    n1=512, n2=0, n3=256,
    d1=1, imfinal1=2, imfinal2=2,
    device=device
).to(device)

model.load_state_dict(torch.load("cvae_gaussian_maps.pth", map_location=device))
model.eval()

# --------------------------
# Generate a single Gaussian sample
# --------------------------
bins = 64
freq_maps, _, _, _ = generate_frequency_maps(bins=bins, seed=42)

# Combine planes in a consistent order
planes_sorted = sorted(freq_maps.keys())
input_array = np.stack([freq_maps[p] for p in planes_sorted], axis=0)
input_tensor = torch.from_numpy(input_array).unsqueeze(0).float().to(device)  # shape (1,15,bins,bins)

# --------------------------
# Run through model
# --------------------------
with torch.no_grad():
    output_tensor, mean, logvar, z = model(input_tensor)

# Convert back to numpy
output_array = output_tensor.squeeze(0).cpu().numpy()  # shape (15, bins, bins)

# --------------------------
# Visualize one plane
# --------------------------
plane_idx = 0  # pick first plane (sorted alphabetically)
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.imshow(input_array[plane_idx], origin='lower', cmap='inferno')
plt.title("Input plane")
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(output_array[plane_idx], origin='lower', cmap='inferno')
plt.title("Reconstruction")
plt.colorbar()
plt.show()

# --------------------------
# Optional: compute MSE for that sample
# --------------------------
mse = ((input_array - output_array)**2).mean()
print(f"MSE for this sample: {mse:.6f}")
