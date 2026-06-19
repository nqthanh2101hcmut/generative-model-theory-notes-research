---
title: "Diffusion and Score-Based Generative Models"
subtitle: "Score Matching, Reverse SDEs, Probability-Flow ODEs, and Sampling Error"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# Abstract

Diffusion and score-based generative models transform data into noise through a forward stochastic process and then learn to reverse this process. Their mathematical foundation involves score functions, denoising score matching, stochastic differential equations, Fokker-Planck equations, reverse-time dynamics, and numerical sampling. This report presents the main objects and highlights research questions concerning score approximation, statistical error, optimization error, and sampling error.

# 1. Forward Noising Process

A diffusion model defines a forward process that gradually corrupts data into noise. In continuous time, a common formulation is an SDE

\[
dX_t=f(X_t,t)dt+g(t)dW_t,
\qquad t\in[0,T],
\]

where \(W_t\) is Brownian motion, \(f\) is a drift, and \(g\) is a diffusion coefficient. The marginal density of \(X_t\) is denoted by \(p_t\).

The goal is to choose the forward process so that \(p_T\) is close to a simple reference distribution, often a standard Gaussian.

# 2. Score Function

The score function of a density \(p_t\) is

\[
s_t(x)=\nabla_x\log p_t(x).
\]

Score-based generative modeling trains a neural network \(s_\theta(x,t)\) to approximate \(s_t(x)\). The learned score is then used to reverse the noising process.

# 3. Reverse-Time SDE

Under appropriate regularity assumptions, the reverse-time dynamics can be written as

\[
dX_t=
\left[f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)\right]dt
+g(t)d\bar W_t,
\]

where time is run backward and \(\bar W_t\) is a reverse-time Brownian motion. In practice, the true score is replaced by the learned score \(s_\theta\).

Thus the sampler depends critically on the accuracy of score estimation.

# 4. Fokker-Planck Equation

The density \(p_t\) of the forward SDE satisfies a Fokker-Planck equation:

\[
\partial_t p_t
=
-\nabla\cdot(f p_t)
+
\frac12 g(t)^2\Delta p_t.
\]

This PDE viewpoint is important because it connects stochastic processes, density evolution, and diffusion-based generative modeling.

# 5. Probability-Flow ODE

For certain diffusion models, there exists a deterministic probability-flow ODE with the same time marginals:

\[
\frac{dX_t}{dt}
=
f(X_t,t)-\frac12 g(t)^2\nabla_x\log p_t(X_t).
\]

This ODE provides a deterministic sampler and connects diffusion models to continuous normalizing flows.

# 6. Denoising Score Matching

Directly estimating \(\nabla \log p_t\) is difficult. Denoising score matching uses perturbed data. If

\[
X_t=\alpha_t X_0+\sigma_t Z,
\qquad Z\sim\mathcal N(0,I),
\]

then the conditional score is often available:

\[
\nabla_{x_t}\log p(x_t|x_0)
=
-\frac{x_t-\alpha_t x_0}{\sigma_t^2}.
\]

The model is trained by minimizing

\[
\mathcal L(\theta)
=
\mathbb E_{t,x_0,z}
\left[
\lambda(t)
\left\|s_\theta(x_t,t)
+
\frac{x_t-\alpha_t x_0}{\sigma_t^2}
\right\|^2
\right].
\]

# 7. Error Sources

Diffusion theory must account for several errors:

1. score approximation error: the neural network class is limited;
2. statistical error: training uses finite data;
3. optimization error: empirical loss is not minimized exactly;
4. time-discretization error: the reverse SDE/ODE is solved numerically;
5. initialization error: terminal distribution may not be exactly Gaussian;
6. model misspecification: assumptions on smoothness or support may fail.

A full guarantee should control how these errors propagate to the generated distribution.

# 8. Research-Level Error Decomposition

A typical objective is to prove a bound of the form

\[
d(\rho_T^\theta,\mu)
\le
C_1\varepsilon_{\mathrm{score}}
+
C_2\varepsilon_{\mathrm{stat}}
+
C_3\varepsilon_{\mathrm{opt}}
+
C_4\varepsilon_{\mathrm{disc}}
+
C_5\varepsilon_{\mathrm{init}},
\]

where \(d\) may be total variation, Wasserstein distance, or another distributional metric. The constants and rates depend on regularity, dimension, noise schedule, and numerical solver.

# 9. Diffusion Alignment as Distributional Optimization

In alignment problems, one may start from a base diffusion model with generated distribution \(\rho_0\), and seek an improved distribution \(\rho\) minimizing

\[
\mathbb E_\rho[\ell(x)]+
\lambda\mathrm{KL}(\rho\|\rho_0).
\]

This formulation separates distributional preference optimization from neural-network parameterization. The mathematical questions include existence, KKT conditions, dual formulations, and sampling from the optimized distribution.

# 10. Research Directions

Important directions include:

1. statistical guarantees for score matching without exact empirical minimization;
2. Wasserstein convergence under non-smooth data distributions;
3. score estimation on low-dimensional manifolds;
4. primal-dual constrained diffusion alignment;
5. sampling error bounds for probability-flow ODE solvers;
6. stability of the generated distribution under perturbations of the score.

# 11. Exercises and Projects

1. Derive the conditional Gaussian score formula.
2. Simulate an Ornstein-Uhlenbeck process.
3. Implement Euler-Maruyama for a simple SDE.
4. Train a toy denoising score network on a Gaussian mixture.
5. Compare stochastic reverse-SDE sampling and deterministic probability-flow ODE sampling in a toy setting.
6. Write a research note on optimization, statistical, and sampling errors in diffusion models.


# References

1. S. Boyd and L. Vandenberghe. *Convex Optimization*. Cambridge University Press, 2004.
2. R. T. Rockafellar and R. J.-B. Wets. *Variational Analysis*. Springer, 1998.
3. G. Dal Maso. *An Introduction to Gamma-Convergence*. Birkhauser, 1993.
4. C. Villani. *Optimal Transport: Old and New*. Springer, 2009.
5. G. Peyre and M. Cuturi. *Computational Optimal Transport*. Foundations and Trends in Machine Learning, 2019.
6. A. Hyvarinen. Estimation of non-normalized statistical models by score matching. *JMLR*, 2005.
7. P. Vincent. A connection between score matching and denoising autoencoders. *Neural Computation*, 2011.
8. J. Ho, A. Jain, and P. Abbeel. Denoising diffusion probabilistic models. *NeurIPS*, 2020.
9. Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and B. Poole. Score-based generative modeling through stochastic differential equations. *ICLR*, 2021.
10. Y. Lipman, R. T. Q. Chen, H. Ben-Hamu, M. Nickel, and M. Le. Flow matching for generative modeling. *ICLR*, 2023.
11. A. Tong, K. Fatras, N. Malkin, G. Huguet, Y. Zhang, J. Rector-Brooks, G. Wolf, and Y. Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. 2023/2024.
12. A. Tong, N. Malkin, K. Fatras, L. Atanackovic, Y. Zhang, G. Huguet, G. Wolf, and Y. Bengio. Simulation-free Schrodinger bridges via score and flow matching. 2023/2024.
13. M. Arbel, A. Korba, A. Salim, and A. Gretton. Maximum mean discrepancy gradient flow. 2019.
14. K. Oko, S. Akiyama, and T. Suzuki. Diffusion models are minimax optimal distribution estimators. *ICML*, 2023.
15. T. Kawata, K. Oko, A. Nitanda, and T. Suzuki. Direct distributional optimization for provable alignment of diffusion models. 2025.
