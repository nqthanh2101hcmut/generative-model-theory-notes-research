---
title: "Optimal Transport, Entropy, and Schrodinger Bridges"
subtitle: "Wasserstein Geometry and Entropic Regularization for Generative Modeling"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# Abstract

Optimal transport provides a geometric theory for moving one probability distribution to another. It is central to flow matching, Wasserstein gradient flows, entropic regularization, and Schrodinger bridge generative models. This report reviews the Kantorovich formulation, Wasserstein distances, dynamic transport, entropic optimal transport, Sinkhorn algorithms, and the connection between Schrodinger bridges and stochastic generative modeling.

# 1. The Transport Viewpoint

Generative modeling may be interpreted as transporting a simple distribution into a complex data distribution. Let \(\rho_0\) be a reference distribution and \(\mu\) be the data distribution. The transport problem asks how to move mass from \(\rho_0\) to \(\mu\).

In deterministic transport, one seeks a map \(T\) such that

\[
T_{\sharp}\rho_0=\mu.
\]

In Kantorovich transport, one seeks a coupling \(\pi\in\Pi(\rho_0,\mu)\), where \(\Pi(\rho_0,\mu)\) is the set of joint distributions with marginals \(\rho_0\) and \(\mu\).

# 2. Kantorovich Optimal Transport

Given a cost \(c(x,y)\), the Kantorovich problem is

\[
\inf_{\pi\in\Pi(\rho,\mu)}
\int_{\mathcal X\times\mathcal X} c(x,y)\,d\pi(x,y).
\]

For \(c(x,y)=\|x-y\|^p\), this defines the Wasserstein distance:

\[
W_p(\rho,\mu)
=
\left(
\inf_{\pi\in\Pi(\rho,\mu)}
\int \|x-y\|^p\,d\pi(x,y)
\right)^{1/p}.
\]

Wasserstein distances are sensitive to geometry, unlike KL divergence, which is more sensitive to density ratios and support mismatch.

# 3. Dynamic Optimal Transport

The Benamou-Brenier formulation expresses \(W_2^2\) through a time-dependent density \(\rho_t\) and velocity field \(v_t\):

\[
W_2^2(\rho_0,\rho_1)
=
\inf_{(\rho_t,v_t)}
\int_0^1\int \|v_t(x)\|^2\,d\rho_t(x)dt
\]

subject to the continuity equation

\[
\partial_t\rho_t+\nabla\cdot(\rho_t v_t)=0,
\qquad
\rho_{t=0}=\rho_0,
\quad
\rho_{t=1}=\rho_1.
\]

This formulation is directly related to flow matching, where the goal is to learn a vector field that transports one distribution into another.

# 4. Entropic Optimal Transport

Classical OT can be computationally expensive and statistically unstable. Entropic OT adds an entropy penalty:

\[
\mathrm{OT}_\varepsilon(\rho,
\mu)
=
\inf_{\pi\in\Pi(\rho,\mu)}
\int c(x,y)\,d\pi(x,y)
+
\varepsilon\mathrm{KL}(\pi\|\rho\otimes\mu).
\]

The parameter \(\varepsilon>0\) smooths the problem. As \(\varepsilon\to 0\), entropic OT approaches classical OT under appropriate assumptions. For \(\varepsilon>0\), the problem is often easier to solve numerically using Sinkhorn iterations.

# 5. Sinkhorn Algorithm

For discrete distributions \(a\in\Delta_m\), \(b\in\Delta_n\), and cost matrix \(C\), entropic OT solves

\[
\min_{P\ge 0}
\langle C,P\rangle
+
\varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1)
\]

subject to

\[
P\mathbf 1=b,
\qquad
P^T\mathbf 1=a.
\]

The solution has the scaling form

\[
P^\star=\operatorname{diag}(u)K\operatorname{diag}(v),
\qquad
K_{ij}=\exp(-C_{ij}/\varepsilon).
\]

Sinkhorn iterations alternately update \(u\) and \(v\) to enforce marginal constraints. This algorithm is important for minibatch OT in flow matching.

# 6. OT for Flow Matching

Flow matching requires probability paths connecting noise and data. One simple approach pairs noise samples and data samples independently. However, random pairing may create long or inefficient trajectories. OT pairing chooses couplings that reduce transport cost and can create straighter paths.

Given paired samples \((x_0,x_1)\), a common conditional path is

\[
x_t=(1-t)x_0+t x_1.
\]

The target velocity is

\[
u_t=x_1-x_0.
\]

If the coupling between \(x_0\) and \(x_1\) is chosen by OT, then trajectories may better approximate displacement interpolation and may be easier for a neural vector field to learn.

# 7. Schrodinger Bridge Problem

The Schrodinger bridge problem asks for the most likely stochastic process connecting two endpoint distributions, relative to a reference stochastic process. In path-space form, one solves

\[
\min_{\mathbb P}
\mathrm{KL}(\mathbb P\|\mathbb Q)
\]

subject to

\[
X_0\sim\rho_0,
\qquad
X_1\sim\mu.
\]

Here \(\mathbb Q\) is a reference path measure, often Brownian motion or a diffusion process. This is an entropy-regularized version of optimal transport on path space.

# 8. Connection to Generative Modeling

Schrodinger bridge models combine ideas from diffusion, optimal transport, and stochastic control. They are attractive because they provide stochastic paths between two distributions while retaining an entropy-regularized optimality principle.

The main mathematical objects are:

- path-space KL divergence;
- endpoint marginal constraints;
- stochastic control representation;
- score functions of intermediate marginals;
- forward and backward potentials;
- entropic interpolation.

# 9. Research Directions

Potential research directions include:

1. stability of entropic OT couplings under empirical approximation;
2. Gamma convergence of entropic OT objectives as \(\varepsilon\to 0\);
3. minibatch OT bias in flow matching;
4. primal-dual formulations of constrained Schrodinger bridges;
5. statistical-sampling error bounds for entropy-regularized transport paths.

# 10. Exercises and Projects

1. Implement one-dimensional Wasserstein distance.
2. Implement Sinkhorn iterations for two empirical distributions.
3. Visualize transport plans between Gaussian mixtures.
4. Compare random pairing and OT pairing in a flow-matching toy problem.
5. Derive the dynamic OT formulation for simple Gaussian distributions.
6. Write a note explaining why Schrodinger bridges can be seen as entropic optimal transport on path space.


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
