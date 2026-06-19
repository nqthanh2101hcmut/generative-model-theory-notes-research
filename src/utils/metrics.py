"""Simple metrics for toy generative modeling experiments."""

from __future__ import annotations

import numpy as np


def pairwise_sq_dists(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x**2, axis=1, keepdims=True)
    y2 = np.sum(y**2, axis=1, keepdims=True).T
    return x2 + y2 - 2.0 * x @ y.T


def mmd_rbf(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Biased MMD^2 estimate with RBF kernel."""
    kxx = np.exp(-pairwise_sq_dists(x, x) / (2 * sigma**2)).mean()
    kyy = np.exp(-pairwise_sq_dists(y, y) / (2 * sigma**2)).mean()
    kxy = np.exp(-pairwise_sq_dists(x, y) / (2 * sigma**2)).mean()
    return float(kxx + kyy - 2 * kxy)


def sliced_wasserstein_2d(x: np.ndarray, y: np.ndarray, num_proj: int = 128, seed: int = 0) -> float:
    """Approximate sliced Wasserstein distance for 2D samples."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((num_proj, x.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    values = []
    for d in dirs:
        xs = np.sort(x @ d)
        ys = np.sort(y @ d)
        m = min(len(xs), len(ys))
        values.append(np.mean((xs[:m] - ys[:m]) ** 2))
    return float(np.sqrt(np.mean(values)))
