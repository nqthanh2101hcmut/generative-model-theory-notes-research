# Mathematical Symbols

## Probability

| Symbol | Meaning |
|---|---|
| \(\mathcal{X}\) | state/data space |
| \(\mathcal{P}(\mathcal{X})\) | set of probability measures on \(\mathcal{X}\) |
| \(\mu, \nu, \rho\) | probability measures |
| \(p(x)\) | density function |
| \(X\sim\mu\) | random variable with distribution \(\mu\) |
| \(\mathbb{E}_{X\sim\mu}[f(X)]\) | expectation under \(\mu\) |
| \(T_\#\mu\) | pushforward of \(\mu\) by map \(T\) |
| \(\hat\mu_n\) | empirical distribution |

## Divergences and distances

| Symbol | Meaning |
|---|---|
| \(\mathrm{KL}(\rho\|\mu)\) | Kullback-Leibler divergence |
| \(H(\rho)\) | entropy |
| \(W_p(\mu,\nu)\) | p-Wasserstein distance |
| \(\mathrm{OT}_\varepsilon(\mu,\nu)\) | entropy-regularized optimal transport cost |
| \(\mathrm{MMD}(\mu,\nu)\) | maximum mean discrepancy |

## Optimization

| Symbol | Meaning |
|---|---|
| \(F(\rho)\) | objective functional |
| \(\arg\min F\) | set of minimizers |
| \(\mathcal{L}(x,\lambda)\) | Lagrangian |
| \(\lambda\) | dual variable / Lagrange multiplier |
| \(\partial f\) | subdifferential |
| \(N_C(x)\) | normal cone to set \(C\) at \(x\) |

## Diffusion and flow

| Symbol | Meaning |
|---|---|
| \(s_\theta(x,t)\) | learned score network |
| \(\nabla_x\log p_t(x)\) | score of distribution \(p_t\) |
| \(v_\theta(x,t)\) | learned vector field |
| \(u_t(x)\) | target vector field in flow matching |
| \(W_t\) | Brownian motion |
| \(dX_t=f(X_t,t)dt+g(t)dW_t\) | forward SDE |
| \(\partial_t p_t + \nabla\cdot(p_t v_t)=0\) | continuity equation |
