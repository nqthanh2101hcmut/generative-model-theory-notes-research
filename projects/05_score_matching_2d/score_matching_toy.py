import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.toy import sample_gaussian_mixture
from src.models.mlp import TimeMLP
from src.utils.plotting import scatter2d, plot_loss
from src.utils.metrics import mmd_rbf, sliced_wasserstein_2d


def sample_dsm_batch(batch_size: int, device: torch.device, sigma: float = 0.35):
    x_np = sample_gaussian_mixture(batch_size)
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    eps = torch.randn_like(x)
    xt = x + sigma * eps
    target_score = -(xt - x) / (sigma**2)
    t = torch.full((batch_size, 1), sigma, device=device)
    return xt, t, target_score


@torch.no_grad()
def langevin_sample(model, n: int = 2000, steps: int = 500, step_size: float = 0.005, sigma: float = 0.35, device: str = "cpu"):
    device_t = torch.device(device)
    x = torch.randn(n, 2, device=device_t) * 2.5
    t = torch.full((n, 1), sigma, device=device_t)
    for _ in range(steps):
        score = model(x, t)
        noise = torch.randn_like(x)
        x = x + step_size * score + (2 * step_size) ** 0.5 * noise
    return x.cpu().numpy()


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    torch.manual_seed(0)
    sigma = 0.35
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeMLP(dim=2, hidden=128, depth=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in trange(3000, desc="training denoising score matching"):
        xt, t, target = sample_dsm_batch(512, device, sigma=sigma)
        pred = model(xt, t)
        loss = ((pred - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    samples = langevin_sample(model, n=2000, steps=500, step_size=0.003, sigma=sigma, device=str(device))
    target = sample_gaussian_mixture(2000, seed=321)

    scatter2d(samples, "Langevin samples from learned score", out / "samples.png")
    scatter2d(target, "Target Gaussian mixture", out / "target.png")
    plot_loss(losses, "Denoising score matching loss", out / "loss.png")

    print("MMD:", mmd_rbf(samples, target, sigma=1.0))
    print("Sliced W2:", sliced_wasserstein_2d(samples, target))


if __name__ == "__main__":
    main()
