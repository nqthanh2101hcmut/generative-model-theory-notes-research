import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import trange

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.data.toy import sample_gaussian_mixture
from src.models.mlp import TimeMLP
from src.utils.plotting import scatter2d, plot_loss
from src.utils.metrics import mmd_rbf, sliced_wasserstein_2d


def sample_batch(batch_size: int, device: torch.device):
    x0 = torch.randn(batch_size, 2, device=device)
    x1_np = sample_gaussian_mixture(batch_size)
    x1 = torch.tensor(x1_np, dtype=torch.float32, device=device)
    t = torch.rand(batch_size, 1, device=device)
    xt = (1.0 - t) * x0 + t * x1
    ut = x1 - x0
    return xt, t, ut


@torch.no_grad()
def sample_flow(model: nn.Module, n: int = 2000, steps: int = 80, device: str = "cpu") -> np.ndarray:
    device_t = torch.device(device)
    x = torch.randn(n, 2, device=device_t)
    h = 1.0 / steps
    for k in range(steps):
        t = torch.full((n, 1), k / steps, device=device_t)
        v = model(x, t)
        x = x + h * v
    return x.cpu().numpy()


def main() -> None:
    out = Path(__file__).parent / "outputs"
    out.mkdir(exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeMLP(dim=2, hidden=128, depth=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    losses = []
    for step in trange(2500, desc="training flow matching"):
        xt, t, ut = sample_batch(512, device)
        pred = model(xt, t)
        loss = ((pred - ut) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))

    samples = sample_flow(model, n=2000, steps=100, device=str(device))
    target = sample_gaussian_mixture(2000, seed=123)

    scatter2d(samples, "Generated samples by toy flow matching", out / "generated.png")
    scatter2d(target, "Target Gaussian mixture", out / "target.png")
    plot_loss(losses, "Flow matching training loss", out / "loss.png")

    print("MMD:", mmd_rbf(samples, target, sigma=1.0))
    print("Sliced W2:", sliced_wasserstein_2d(samples, target))


if __name__ == "__main__":
    main()
