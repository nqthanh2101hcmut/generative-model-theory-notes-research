---
title: "Flow Matching and Continuous Flows"
subtitle: "Vector Fields, Continuity Equations, Optimal Transport Paths, and Stability"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# Abstract

Flow matching trains a neural vector field that transports a reference distribution into a data distribution through an ordinary differential equation. It provides a simulation-free alternative to likelihood-based continuous normalizing flows and a deterministic counterpart to diffusion-based methods. This report reviews flow maps, continuity equations, conditional flow matching, optimal-transport couplings, and theoretical questions about vector-field approximation and distributional stability.

# 1. Continuous Normalizing Flow Viewpoint

Let \(X_0\sim \rho_0\). A time-dependent vector field \(v_t\) defines an ODE

\[
\frac{dX_t}{dt}=v_t(X_t),
\qquad t\in[0,1].
\]

The flow map \(T_t\) sends \(X_0\) to \(X_t\), and the distribution at time \(t\) is

\[
\rho_t=(T_t)_{\sharp}\rho_0.
\]

The generative goal is

\[
\rho_1\approx \mu.
\]

# 2. Continuity Equation

If particles follow the ODE, their density evolves according to the continuity equation

\[
\partial_t\rho_t+
\nabla\cdot(\rho_t v_t)=0.
\]

This PDE is the deterministic analogue of the Fokker-Planck equation for diffusion processes.

# 3. Flow Matching Objective

Flow matching constructs probability paths \(\rho_t\) and target vector fields \(u_t\). A neural network \(v_\theta(x,t)\) is trained by minimizing

\[
\mathcal L(\theta)=
\mathbb E_{t,X_t}
\left[
\|v_\theta(X_t,t)-u_t(X_t)\|^2
\right].
\]

The advantage is that one may train the vector field without simulating the learned ODE during training.

# 4. Conditional Flow Matching

A common construction samples pairs \((X_0,X_1)\) and defines a conditional path

\[
X_t=(1-t)X_0+tX_1.
\]

The conditional velocity is

\[
U_t=X_1-X_0.
\]

The training objective becomes

\[
\mathcal L(\theta)=
\mathbb E_{t,X_0,X_1}
\left[
\|v_\theta(X_t,t)-(X_1-X_0)\|^2
\right].
\]

Different choices of coupling between \(X_0\) and \(X_1\) lead to different probability paths and training difficulty.

# 5. Optimal Transport Couplings

If \((X_0,X_1)\) are paired independently, the paths may be inefficient. If they are paired using an optimal transport plan, the resulting paths can be shorter and straighter.

For a coupling \(\pi\in\Pi(\rho_0,\mu)\), the conditional flow-matching objective is

\[
\mathbb E_{(X_0,X_1)\sim\pi,\ t\sim U[0,1]}
\left[
\|v_\theta((1-t)X_0+tX_1,t)-(X_1-X_0)\|^2
\right].
\]

This connects flow matching with dynamic optimal transport.

# 6. Stability of ODE Flows

Suppose \(v\) is the target vector field and \(\hat v\) is the learned vector field. Let \(X_t\) and \(\hat X_t\) solve

\[
\dot X_t=v_t(X_t),
\qquad
\dot{\hat X}_t=\hat v_t(\hat X_t).
\]

If \(v\) is Lipschitz and \(\hat v\) is close to \(v\), then Gronwall-type estimates give

\[
\|X_t-\hat X_t\|
\le
C\int_0^t
\|v_s(\hat X_s)-\hat v_s(\hat X_s)\|ds.
\]

This indicates how vector-field approximation error propagates into sample error.

# 7. Distributional Stability

The terminal distribution error may be bounded in Wasserstein distance:

\[
W_2(\hat\rho_1,\rho_1)
\le
\left(\mathbb E\|\hat X_1-X_1\|^2\right)^{1/2}.
\]

Thus, controlling pathwise ODE error can imply distributional convergence.

# 8. Numerical Error

In practice, the ODE is solved by Euler or higher-order solvers. Euler discretization gives

\[
X_{k+1}=X_k+h v_\theta(X_k,t_k).
\]

The total error includes both learned-vector-field error and numerical discretization error. Fast generative models seek good samples with very few ODE steps, making numerical stability especially important.

# 9. Research Directions

Important research questions include:

1. how path choice affects vector-field approximation difficulty;
2. how minibatch OT bias affects flow-matching objectives;
3. whether entropy-regularized OT paths are variationally stable;
4. how flow-matching error bounds depend on intrinsic dimension;
5. how one-step or few-step flows trade sample quality for speed;
6. how constrained generation can be formulated through primal-dual flow matching.

# 10. Suggested Project

A concrete project is:

\[
\textbf{Variational Convergence of Minibatch Entropic OT Objectives in Flow Matching.}
\]

The project studies the objective

\[
F_{n,\varepsilon}(v)=
\mathbb E_{(x_0,x_1)\sim\pi_{n,\varepsilon},\ t}
\left[
\|v(x_t,t)-(x_1-x_0)\|^2
\right],
\]

where \(\pi_{n,\varepsilon}\) is an empirical entropic OT coupling. The main theoretical question is whether \(F_{n,\varepsilon}\) converges to an ideal population objective and whether minimizers are stable.

# 11. Exercises and Projects

1. Derive the continuity equation from a particle ODE.
2. Implement flow matching on a two-dimensional Gaussian mixture.
3. Compare independent pairing and OT pairing.
4. Measure terminal sample quality by sliced Wasserstein distance or MMD.
5. Study how solver step size affects sample quality.
6. Write a theorem template for stability of ODE-generated distributions under vector-field perturbation.


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
