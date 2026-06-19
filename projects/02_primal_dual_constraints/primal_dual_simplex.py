from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def normalize(p: np.ndarray) -> np.ndarray:
    p = np.maximum(p, 1e-12)
    return p / p.sum()


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    rng = np.random.default_rng(1)
    d = 20
    c = rng.normal(size=d)
    a = np.linspace(-1.0, 1.0, d)
    b = -0.15
    q = np.ones(d) / d
    lam = 0.2
    eta_p = 0.15
    eta_dual = 0.1
    steps = 600

    p = q.copy()
    alpha = 0.0
    objectives, violations, alphas = [], [], []

    for _ in range(steps):
        violation = float(np.dot(p, a) - b)
        grad = c + alpha * a + lam * (np.log(p + 1e-12) - np.log(q + 1e-12) + 1.0)

        # Entropic mirror descent / exponentiated gradient update.
        p = normalize(p * np.exp(-eta_p * grad))

        # Projected dual ascent.
        alpha = max(0.0, alpha + eta_dual * violation)

        kl = np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12)))
        obj = float(np.dot(p, c) + lam * kl)
        objectives.append(obj)
        violations.append(max(0.0, violation))
        alphas.append(alpha)

    print("final alpha:", alpha)
    print("constraint value <p,a>:", np.dot(p, a), "threshold b:", b)
    print("final p:", np.round(p, 3))

    plt.figure(figsize=(6, 4))
    plt.plot(objectives)
    plt.xlabel("step")
    plt.ylabel("objective")
    plt.title("Primal objective")
    plt.tight_layout()
    plt.savefig(out / "objective.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(violations)
    plt.xlabel("step")
    plt.ylabel("positive violation")
    plt.title("Constraint violation")
    plt.tight_layout()
    plt.savefig(out / "constraint_violation.png", dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(np.arange(d), p)
    plt.xlabel("state")
    plt.ylabel("probability")
    plt.title("Final distribution")
    plt.tight_layout()
    plt.savefig(out / "final_distribution.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
