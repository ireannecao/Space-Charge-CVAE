# generate_frequency_maps_dataset.py
import numpy as np
from beam_data import generate_frequency_maps, PLANE_MAP

def generate_and_save_frequency_maps(filename="frequency_maps.npz", n_samples=10000, bins=64, seed=None):
    """
    Generate N Gaussian frequency maps and save them to a .npz file.
    Each sample contains all 15 planes stacked as (15, bins, bins).
    """
    rng = np.random.default_rng(seed)
    data = []

    for i in range(n_samples):
        # optionally vary seed per sample
        sample_seed = None if seed is None else int(rng.integers(0, 1e6))
        freq_maps, _, _, _ = generate_frequency_maps(bins=bins, seed=sample_seed)
        planes_sorted = sorted(freq_maps.keys())
        arr = np.stack([freq_maps[p] for p in planes_sorted], axis=0)  # shape (15, bins, bins)
        data.append(arr)

    data = np.stack(data, axis=0)  # shape (N, 15, bins, bins)
    np.savez_compressed(filename, data=data)
    print(f"✅ Saved {n_samples} frequency maps to {filename}")


if __name__ == "__main__":
    generate_and_save_frequency_maps(filename="frequency_maps.npz", n_samples=10000, bins=64, seed=42)
