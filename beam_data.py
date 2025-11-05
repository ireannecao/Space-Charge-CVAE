import numpy as np
import matplotlib.pyplot as plt
import time

# --------------------------
# 1. Full 15 Projection Map
# --------------------------
PLANE_MAP = {
    # Position–Position
    "x-y": (0, 1),
    "x-z": (0, 2),
    "y-z": (1, 2),

    # Position–Momentum
    "x-px": (0, 3),
    "x-py": (0, 4),
    "x-pz": (0, 5),
    "y-px": (1, 3),
    "y-py": (1, 4),
    "y-pz": (1, 5),
    "z-px": (2, 3),
    "z-py": (2, 4),
    "z-pz": (2, 5),

    # Momentum–Momentum
    "px-py": (3, 4),
    "px-pz": (3, 5),
    "py-pz": (4, 5),
}

# --------------------------
# 2. Covariance Matrix Utils
# --------------------------
def build_covariance(sigmas=None, corrs=None, generate_random=False, seed=None):
    """
    Build a full 6x6 covariance matrix.
    - If generate_random=True, produce a random positive definite covariance.
    """
    dim = 6
    if generate_random:
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(dim, dim))
        Sigma = A @ A.T
        if sigmas is None:
            sigmas = rng.uniform(0.5, 2.0, size=dim)
        D = np.diag(sigmas / np.sqrt(np.diag(Sigma)))
        return D @ Sigma @ D
    else:
        sigmas = np.array(sigmas)
        if corrs is None:
            corrs = np.zeros(dim*(dim-1)//2)
        Sigma = np.diag(sigmas**2)
        idx = 0
        for i in range(dim):
            for j in range(i+1, dim):
                Sigma[i, j] = corrs[idx]*sigmas[i]*sigmas[j]
                Sigma[j, i] = Sigma[i, j]
                idx += 1
        return Sigma

# --------------------------
# 3. Analytic Gaussian Density
# --------------------------
def gaussian_2d_density(X, Y, Sigma_2x2):
    """
    Compute a smooth, normalized 2D Gaussian density on grid (X,Y)
    with covariance Sigma_2x2 (zero mean).
    """
    inv_cov = np.linalg.inv(Sigma_2x2)
    det_cov = np.linalg.det(Sigma_2x2)
    pos = np.stack([X, Y], axis=-1)
    exp_term = np.einsum("...k,kl,...l->...", pos, inv_cov, pos)
    pdf = np.exp(-0.5 * exp_term) / (2 * np.pi * np.sqrt(det_cov))
    return pdf / pdf.sum()

# --------------------------
# 4. Frequency Map Generator
# --------------------------
def generate_frequency_maps(bins=128, planes=PLANE_MAP, Sigma=None,
                             correlated=True, seed=None):
    """
    Generate analytic, smooth 2D Gaussian frequency maps for each projection.
    All maps share the same coordinate range and resolution.
    """
    if planes is None:
        planes = list(PLANE_MAP.keys())

    dim = 6
    rng = np.random.default_rng(seed)

    # Build covariance
    if Sigma is None:
        Sigma = build_covariance(generate_random=correlated, seed=seed)

    # Common coordinate range and grid
    global_min, global_max = -4, 4
    x = np.linspace(global_min, global_max, bins)
    y = np.linspace(global_min, global_max, bins)
    X, Y = np.meshgrid(x, y)

    freq_maps = {}
    for plane in planes:
        i, j = PLANE_MAP[plane]
        Sigma_2x2 = Sigma[np.ix_([i, j], [i, j])]
        freq_maps[plane] = gaussian_2d_density(X, Y, Sigma_2x2)

    all_values = np.concatenate([f.ravel() for f in freq_maps.values()])
    vmin, vmax = all_values.min(), all_values.max()
    for plane in planes:
        freq_maps[plane] = (freq_maps[plane] - vmin) / (vmax - vmin)

    return freq_maps, Sigma, X, Y

# --------------------------
# 5. Visualization (3x5 Grid, labeled axes)
# --------------------------
def visualize_all_frequency_maps(frequency_maps, planes=None,
                                 X=None, Y=None, cmap="inferno", log_scale=True):
    """
    Visualize 2D frequency maps for all (or selected) planes in a 3x5 grid.
    Adds axis labels and numeric tick ranges for clarity.
    """
    if planes is None:
        planes = list(PLANE_MAP.keys())

    n_rows, n_cols = 3, 5
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5*n_cols, 4.5*n_rows))
    axs = axs.flatten()

    # infer coordinate ranges if available
    if X is not None and Y is not None:
        x_min, x_max = X[0, 0], X[0, -1]
        y_min, y_max = Y[0, 0], Y[-1, 0]
    else:
        x_min, x_max, y_min, y_max = -4, 4, -4, 4

    for i, plane in enumerate(planes):
        ax = axs[i]
        F = frequency_maps[plane]
        display = np.log1p(F) if log_scale else F

        im = ax.imshow(display, origin="lower", cmap=cmap, aspect="equal",
                       extent=[x_min, x_max, y_min, y_max])

        # split plane name for axis labels, e.g. "x-px" → x-label="x", y-label="px"
        x_label, y_label = plane.split("-")
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.set_title(plane, fontsize=11)
        ax.tick_params(labelsize=8)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for j in range(len(planes), len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()

# --------------------------
# 6. Main Demo
# --------------------------
if __name__ == "__main__":
    bins = 128
    planes = list(PLANE_MAP.keys())

    start = time.time()
    freq_maps, Sigma, X, Y = generate_frequency_maps(bins=bins, planes=planes, seed=42)
    elapsed = time.time() - start

    print(f"Generated analytic Gaussian frequency maps in {elapsed:.2f} s")
    print("Covariance matrix:\n", np.round(Sigma, 3))

    visualize_all_frequency_maps(freq_maps, planes, X, Y)

    # Benchmark loop
    n_trials = 1000
    start = time.time()
    for _ in range(n_trials):
        generate_frequency_maps(bins=bins, planes=planes, seed=None)
    elapsed = time.time() - start
    print(f"\nBenchmark: {n_trials} calls to generate_frequency_maps")
    print(f"Total time: {elapsed:.2f} s")
    print(f"Average per call: {elapsed/n_trials:.6f} s")
