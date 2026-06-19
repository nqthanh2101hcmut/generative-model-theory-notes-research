"""Toy data distributions for generative model theory experiments."""

from __future__ import annotations

import math
import numpy as np


def sample_gaussian_mixture(n: int, dim: int = 2, seed: int | None = None) -> np.ndarray:
    """Sample from a simple 2D Gaussian mixture."""
    rng = np.random.default_rng(seed)
    if dim != 2:
        raise ValueError("This toy mixture is implemented for dim=2 only.")
    centers = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]])
    idx = rng.integers(0, len(centers), size=n)
    return centers[idx] + 0.35 * rng.standard_normal((n, 2))


def sample_two_moons(n: int, noise: float = 0.08, seed: int | None = None) -> np.ndarray:
    """Sample a lightweight two-moons dataset without sklearn."""
    rng = np.random.default_rng(seed)
    n1 = n // 2
    n2 = n - n1
    theta1 = rng.uniform(0, math.pi, size=n1)
    theta2 = rng.uniform(0, math.pi, size=n2)
    moon1 = np.stack([np.cos(theta1), np.sin(theta1)], axis=1)
    moon2 = np.stack([1.0 - np.cos(theta2), 0.5 - np.sin(theta2)], axis=1)
    x = np.concatenate([moon1, moon2], axis=0)
    x += noise * rng.standard_normal(x.shape)
    x = 2.0 * (x - x.mean(axis=0, keepdims=True))
    return x


def sample_standard_normal(n: int, dim: int = 2, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim))
