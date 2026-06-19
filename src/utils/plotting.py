"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def scatter2d(x: np.ndarray, title: str, path: str | Path, s: int = 8) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=s, alpha=0.7)
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_loss(losses: list[float], title: str, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    plt.figure(figsize=(6, 4))
    plt.plot(losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
