import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.toy import sample_gaussian_mixture, sample_standard_normal
from src.utils.metrics import pairwise_sq_dists


def sinkhorn(a: np.ndarray, b: np.ndarray, C: np.ndarray, eps: float = 0.2, n_iter: int = 500):
    K = np.exp(-C / eps)
    u = np.ones_like(a)
    v = np.ones_like(b)
    for _ in range(n_iter):
        u = a / (K @ v + 1e-12)
        v = b / (K.T @ u + 1e-12)
    P = np.diag(u) @ K @ np.diag(v)
    return P


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    n = 80
    x = sample_standard_normal(n, dim=2, seed=0)
    y = sample_gaussian_mixture(n, dim=2, seed=1)
    a = np.ones(n) / n
    b = np.ones(n) / n
    C = pairwise_sq_dists(x, y)
    P = sinkhorn(a, b, C, eps=0.5)

    plt.figure(figsize=(5, 5))
    plt.scatter(x[:, 0], x[:, 1], s=15, label="source")
    plt.scatter(y[:, 0], y[:, 1], s=15, label="target")
    plt.legend()
    plt.axis("equal")
    plt.title("Source and target samples")
    plt.tight_layout()
    plt.savefig(out / "samples.png", dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.imshow(P, aspect="auto")
    plt.colorbar(label="mass")
    plt.title("Entropic OT transport plan")
    plt.xlabel("target index")
    plt.ylabel("source index")
    plt.tight_layout()
    plt.savefig(out / "transport_plan.png", dpi=160)
    plt.close()

    print("transport cost:", float(np.sum(P * C)))
    print("marginal error source:", np.linalg.norm(P.sum(axis=1) - a))
    print("marginal error target:", np.linalg.norm(P.sum(axis=0) - b))


if __name__ == "__main__":
    main()
