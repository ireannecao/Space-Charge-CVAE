# beam_data.py
import numpy as np
from scipy.stats import qmc, norm
import matplotlib.pyplot as plt
import time

# --------------------------
# 1. Latin Hypercube Sampler for beam parameters
# --------------------------
def lhs_beam_params(n_samples, sigma_ranges, corr_ranges=None, seed=None):
    """
    Generate beam parameters (sigmas and optional correlations) using LHS.
    Returns (sigmas, corrs)
    """
    dim = len(sigma_ranges) + (len(corr_ranges) if corr_ranges else 0)
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    lhs = sampler.random(n=n_samples)

    params = []
    ranges = sigma_ranges + (corr_ranges if corr_ranges else [])
    for j, (low, high) in enumerate(ranges):
        col = qmc.scale(lhs[:, [j]], low, high)  # keep 2D
        params.append(col)

    params = np.hstack(params)  # shape (n_samples, dim)

    sigmas = params[:, :len(sigma_ranges)]
    corrs = params[:, len(sigma_ranges):] if corr_ranges else np.zeros((n_samples, 0))

    return sigmas, corrs

# --------------------------
# 2. Covariance Matrix Utils (always positive definite)
# --------------------------
def build_covariance(sigmas=None, corrs=None, generate_random=False, seed=None):
    """
    Build a full 6x6 covariance matrix.
    - If generate_random=True, always produce a positive definite 6x6 covariance.
    """
    dim = 6

    if generate_random:
        rng = np.random.default_rng(seed)
        # 1. Generate a random Gaussian matrix
        A = rng.normal(size=(dim, dim))
        Sigma = A @ A.T  # symmetric positive definite

        # 2. Scale diagonals to desired sigma range
        if sigmas is None:
            sigmas = rng.uniform(0.5, 2.0, size=dim)
        D = np.diag(sigmas / np.sqrt(np.diag(Sigma)))
        Sigma = D @ Sigma @ D

        return Sigma
    else:
        sigmas = np.array(sigmas)
        if corrs is None:
            corrs = np.zeros(dim*(dim-1)//2)
        Sigma = np.diag(sigmas**2)
        idx = 0
        for i in range(dim):
            for j in range(i+1, dim):
                Sigma[i,j] = corrs[idx]*sigmas[i]*sigmas[j]
                Sigma[j,i] = Sigma[i,j]
                idx += 1
        return Sigma

def sample_gaussian_from_cov(Sigma, n_samples, method="halton", seed=None):
    """
    Generate samples from a multivariate Gaussian with covariance Sigma.
    method: "random", "halton", "lhs"
    """
    dim = Sigma.shape[0]
    rng = np.random.default_rng(seed)

    if method == "random":
        mean = np.zeros(dim)
        return rng.multivariate_normal(mean, Sigma, size=n_samples)

    elif method == "halton":
        sampler = qmc.Halton(d=dim, seed=seed)
        u = sampler.random(n_samples)
        z = norm.ppf(u)

    elif method == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        u = sampler.random(n_samples)
        z = norm.ppf(u)

    else:
        raise ValueError("method must be 'random', 'halton', 'lhs', or 'hammersley'")

    # Apply covariance via Cholesky
    L = np.linalg.cholesky(Sigma)
    x = z @ L.T
    return x

# --------------------------
# 3. Plane mapping
# --------------------------
PLANE_MAP = {
    "x-px": (0, 3), "y-py": (1, 4), "z-pz": (2, 5),
    "x-y": (0, 1), "x-z": (0, 2), "y-z": (1, 2),
    "px-py": (3, 4), "px-pz": (3, 5), "py-pz": (4, 5),
}

# --------------------------
# 4. Distribution Generator
# --------------------------
def generate_distribution_lhs(n_samples=5000, planes=None,
                              sigma_ranges=None, corr_ranges=None,
                              seed=None, save_path=None, method="halton",
                              correlated=True):
    """
    Generate one fake distribution: Gaussian samples with optional correlations.
    Returns (samples, Sigma)
    """
    if planes is None:
        planes = ["x-px", "y-py", "z-pz"]

    if correlated:
        Sigma = build_covariance(generate_random=True, seed=seed)
    else:
        if sigma_ranges is None:
            sigma_ranges = [(0.5, 2.0)] * 6
        if corr_ranges is None:
            corr_ranges = []
        sigmas, corrs = lhs_beam_params(1, sigma_ranges, corr_ranges, seed=seed)
        Sigma = build_covariance(sigmas[0], corrs[0] if corrs.shape[1]>0 else None)

    samples = sample_gaussian_from_cov(Sigma, n_samples, method=method, seed=seed)

    if save_path is not None:
        np.save(save_path, samples)

    return samples, Sigma

# --------------------------
# 5. Visualization
# --------------------------
def visualize_all_distributions_from_samples(samples_list, planes, Sigmas_list=None, bins=64, log_scale=True, normalize=True, cmap="inferno"):
    """
    Visualize 2D histograms of multiple distributions.
    """
    n_dists = len(samples_list)
    C = len(planes)

    # Compute global min/max per plane
    global_min = [np.inf]*C
    global_max = [-np.inf]*C
    global_max_count = [0]*C
    for i, plane in enumerate(planes):
        idx0, idx1 = PLANE_MAP[plane]
        for samples in samples_list:
            x = samples[:, idx0]
            y = samples[:, idx1]
            global_min[i] = min(global_min[i], x.min(), y.min())
            global_max[i] = max(global_max[i], x.max(), y.max())

    # Convert samples to histograms
    hists_list = []
    for samples in samples_list:
        hists = []
        for i, plane in enumerate(planes):
            idx0, idx1 = PLANE_MAP[plane]
            H, xedges, yedges = np.histogram2d(samples[:, idx0], samples[:, idx1],
                                               bins=bins, range=[[global_min[i], global_max[i]], [global_min[i], global_max[i]]])
            H = H.T
            if normalize:
                H = H / H.sum()
            hists.append(H)
            global_max_count[i] = max(global_max_count[i], H.max())
        hists_list.append(hists)

    # Plot all distributions
    fig, axs = plt.subplots(n_dists, C, figsize=(5*C, 4*n_dists))
    if n_dists == 1: axs = [axs]
    if C == 1: axs = [[ax] for ax in axs]

    for d in range(n_dists):
        if Sigmas_list is not None:
            print(f"Distribution {d+1} covariance matrix:\n{Sigmas_list[d]}\n")
            sigmas = np.round(np.sqrt(np.diag(Sigmas_list[d])), 3)
            sig_str = ", ".join(map(str, sigmas))
        else:
            sig_str = ""

        for i in range(C):
            display = np.log1p(hists_list[d][i]) if log_scale else hists_list[d][i]
            im = axs[d][i].imshow(display, origin="lower", cmap=cmap,
                                   extent=[global_min[i], global_max[i], global_min[i], global_max[i]],
                                   vmin=0, vmax=global_max_count[i])
            title = f"Dist {d+1} - {planes[i]}"
            if sig_str:
                title += f"\nσ: [{sig_str}]"
            axs[d][i].set_title(title)
            plt.colorbar(im, ax=axs[d][i])

    plt.tight_layout()
    plt.show()

# --------------------------
# 6. Main Demo
# --------------------------
if __name__ == "__main__":
    n_samples = 5000
    planes = ["x-px", "y-py", "z-pz"]

    # Generate multiple distributions
    samples_list, Sigmas_list = [], []
    for seed in [1, 2, 3]:
        samples, Sigma = generate_distribution_lhs(n_samples=n_samples, planes=planes, seed=seed, correlated=True)
        samples_list.append(samples)
        Sigmas_list.append(Sigma)

    # Visualize
    visualize_all_distributions_from_samples(samples_list, planes, Sigmas_list, bins=64, log_scale=True, normalize=True)

    # Benchmark
    n_dists = 100
    start = time.time()
    for _ in range(n_dists):
        generate_distribution_lhs(n_samples=n_samples, planes=planes)
    elapsed = time.time() - start
    print(f"Generated {n_dists} distributions in {elapsed:.2f} seconds "
          f"({elapsed/n_dists:.4f} sec per distribution)")
