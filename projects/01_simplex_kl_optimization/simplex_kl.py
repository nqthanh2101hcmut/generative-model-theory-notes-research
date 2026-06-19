from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def kl_regularized_solution(cost: np.ndarray, q: np.ndarray, lam: float) -> np.ndarray:
    logits = np.log(q + 1e-12) - cost / lam
    logits -= logits.max()
    p = np.exp(logits)
    return p / p.sum()


def objective(p: np.ndarray, cost: np.ndarray, q: np.ndarray, lam: float) -> float:
    kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))
    return float(np.dot(p, cost) + lam * kl)


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    d = 10
    rng = np.random.default_rng(0)
    cost = rng.normal(size=d)
    q = np.ones(d) / d
    lam = 0.4

    p_star = kl_regularized_solution(cost, q, lam)

    print("cost:", np.round(cost, 3))
    print("q:", np.round(q, 3))
    print("p_star:", np.round(p_star, 3))
    print("objective:", objective(p_star, cost, q, lam))

    x = np.arange(d)
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, q, width=0.4, label="reference q")
    plt.bar(x + 0.2, p_star, width=0.4, label="optimal p")
    plt.xlabel("state")
    plt.ylabel("probability")
    plt.title("KL-regularized distributional optimization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "simplex_kl_solution.png", dpi=160)


if __name__ == "__main__":
    main()
