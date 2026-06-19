from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    steps = np.arange(1, 301)
    n = 1000
    width = 128
    solver_steps = 80

    optimization = 1.5 / np.sqrt(steps)
    statistical = np.ones_like(steps) * (1.0 / np.sqrt(n))
    approximation = np.ones_like(steps) * (1.0 / np.sqrt(width))
    sampling = np.ones_like(steps) * (1.0 / solver_steps)
    total = optimization + statistical + approximation + sampling

    plt.figure(figsize=(7, 4))
    plt.plot(steps, optimization, label="optimization")
    plt.plot(steps, statistical, label="statistical")
    plt.plot(steps, approximation, label="approximation")
    plt.plot(steps, sampling, label="sampling")
    plt.plot(steps, total, label="total", linewidth=2)
    plt.xlabel("training step")
    plt.ylabel("toy error")
    plt.title("Toy error decomposition")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "error_decomposition.png", dpi=160)
    plt.close()

    print("This is a template only. Replace toy curves with measured/proved bounds.")


if __name__ == "__main__":
    main()
