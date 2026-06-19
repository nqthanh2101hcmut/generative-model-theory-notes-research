---
title: "Research Program and Paper Plan"
subtitle: "A PhD-Oriented Agenda for Optimization and Statistical Foundations of Generative Models"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# Abstract

This report proposes a PhD-level research program at the intersection of optimization, variational analysis, probability, statistics, optimal transport, and generative modeling. The guiding theme is entropy-regularized distributional optimization with applications to diffusion models, flow matching, and generative-model alignment. The report outlines research questions, possible theorems, experiments, and publication pathways.

# 1. Research Theme

The proposed research theme is:

\[
\textbf{Optimization and Variational Stability of Entropy-Regularized Distributional Problems for Diffusion and Flow Generative Models.}
\]

A mathematically conservative title for an applied mathematics PhD is:

\[
\textbf{Variational Convergence, Optimality Conditions, and Primal-Dual Algorithms for Entropy-Regularized Optimization Problems on Probability Spaces.}
\]

A machine-learning-facing title is:

\[
\textbf{Optimization, Statistical, and Sampling Foundations of Diffusion and Flow-Based Generative Models.}
\]

# 2. Core Mathematical Problem

The core problem is

\[
\min_{\rho\in\mathcal P(\mathcal X)}
F(\rho)
=
\mathbb E_\rho[\ell(x)]
+
\lambda\mathrm{KL}(\rho\|\rho_0)
+
\gamma\mathrm{OT}_\varepsilon(\rho,
\mu)
\]

subject to

\[
\mathbb E_\rho[c_j(x)]\le b_j,
\qquad j=1,\ldots,m.
\]

This formulation is broad enough to include distributional alignment, entropy-regularized transport, constrained generative modeling, and empirical approximations.

# 3. Thesis Structure

A possible thesis can be organized around three pillars.

## Pillar 1: Variational Analysis and Optimality

Questions:

- Does the objective admit minimizers?
- Are minimizers unique under entropy regularization?
- Can KKT conditions be derived?
- Does strong duality hold?
- Are minimizers stable under perturbations of \(\ell\), \(\mu\), or \(\rho_0\)?

Expected results:

- existence theorem;
- KKT theorem;
- duality theorem;
- stability theorem;
- Gamma/epi convergence theorem.

## Pillar 2: Algorithms and Error Bounds

Questions:

- Can mirror descent or primal-dual methods solve the problem?
- Can one bound the primal-dual gap?
- Can one control constraint violation?
- How do empirical and minibatch approximations affect convergence?

Expected results:

- primal-dual algorithm;
- convergence rate;
- constraint violation bound;
- empirical approximation bound;
- numerical demonstrations.

## Pillar 3: Applications to Diffusion and Flow Models

Questions:

- How does the distributional formulation apply to diffusion alignment?
- How can flow matching be interpreted through OT and variational stability?
- How do score or vector-field errors propagate into distributional errors?

Expected results:

- toy diffusion-alignment experiment;
- flow-matching experiment with OT pairing;
- error decomposition for generated distributions;
- research note linking theory and implementation.

# 4. Paper Plan

## Paper 1: Variational Stability and KKT Conditions

Provisional title:

\[
\textit{KKT Conditions and Variational Stability for Entropy-Regularized Distributional Optimization.}
\]

Target contribution:

- rigorous formulation on probability spaces;
- existence of minimizers;
- KKT conditions under constraint qualification;
- stability of minimizers under perturbation;
- finite-dimensional examples.

Potential venues:

- Journal of Optimization Theory and Applications;
- Applied Mathematics and Optimization;
- Set-Valued and Variational Analysis;
- Optimization.

## Paper 2: Primal-Dual Algorithms

Provisional title:

\[
\textit{Primal-Dual Algorithms for Constrained Entropy-Regularized Distributional Optimization.}
\]

Target contribution:

- KL mirror descent primal updates;
- projected dual ascent;
- primal-dual gap bounds;
- constraint violation analysis;
- reproducible numerical experiments.

Potential venues:

- Optimization Methods and Software;
- Computational Optimization and Applications;
- Journal of Optimization Theory and Applications.

## Paper 3: Generative Model Application

Provisional title:

\[
\textit{Optimization, Statistical, and Sampling Error Decomposition for Diffusion and Flow-Based Generative Models.}
\]

Target contribution:

- distributional optimization formulation;
- score/vector-field approximation analysis;
- empirical approximation error;
- sampling and discretization error;
- toy diffusion and flow experiments.

Potential venues:

- Information and Inference;
- Machine Learning;
- Transactions on Machine Learning Research;
- AISTATS, UAI, or ICLR workshop.

# 5. Theorem Roadmap

A minimal theorem roadmap is as follows.

## Theorem A: Existence

Assume \(\mathcal X\) is compact, \(\ell\) and \(c_j\) are lower semicontinuous, and \(\lambda>0\). Show that the feasible entropy-regularized problem admits at least one minimizer.

## Theorem B: KKT Conditions

Under convexity and a Slater-type condition, show that there exist multipliers \(\alpha_j\ge 0\) such that \((\rho^\star,\alpha^\star)\) satisfies stationarity, primal feasibility, dual feasibility, and complementary slackness.

## Theorem C: Variational Stability

Let \(F_n\) be empirical objectives. If \(F_n\) Gamma-converges to \(F\) and the family is equicoercive, then every cluster point of minimizers of \(F_n\) is a minimizer of \(F\). If the minimizer is unique, the full sequence converges.

## Theorem D: Primal-Dual Gap

For a suitable primal-dual mirror-descent method, prove a bound of the form

\[
\mathrm{Gap}(\bar\rho_K,\bar\alpha_K)
\le
O(K^{-1/2})
\]

under bounded-gradient assumptions.

## Theorem E: Generative Error Decomposition

For a learned score or vector field, prove a bound of the form

\[
d(\rho_1^\theta,\mu)
\le
C_1\varepsilon_{\mathrm{approx}}
+C_2\varepsilon_{\mathrm{stat}}
+C_3\varepsilon_{\mathrm{opt}}
+C_4\varepsilon_{\mathrm{sample}}.
\]

# 6. Experimental Program

The experiments should be simple, reproducible, and aligned with the mathematical theory.

## Experiment 1: KL-Regularized Simplex Optimization

Goal: demonstrate Gibbs-form solutions and stability under cost perturbations.

## Experiment 2: Constrained Primal-Dual Optimization

Goal: show convergence of constraint violation and objective value.

## Experiment 3: Sinkhorn OT and Coupling Stability

Goal: visualize entropic OT plans under different \(\varepsilon\).

## Experiment 4: Flow Matching with Random Pairing vs OT Pairing

Goal: compare trajectory straightness, vector-field loss, and terminal distribution quality.

## Experiment 5: Score Matching on Gaussian Mixtures

Goal: visualize learned score fields and sampling trajectories.

# 7. Risk Management

The main risks are:

1. the problem becomes too abstract and loses connection to generative models;
2. the problem becomes too engineering-driven and loses mathematical depth;
3. theoretical assumptions become too strong;
4. experiments require more computational resources than available.

A safe strategy is to keep the first paper mathematically focused, the second algorithmic, and the third connected to generative models.

# 8. Research Identity

For an academic CV:

> Research interests: Variational analysis and optimization on probability spaces; entropy-regularized distributional optimization; optimality conditions, duality, and stability; probability, statistics, and optimal transport for diffusion and flow-based generative models.

For an ML-facing profile:

> Research interests: Optimization and statistical foundations of generative models, with a focus on diffusion models, flow matching, optimal transport, and entropy-regularized distributional optimization.

# 9. Final Recommendation

The recommended PhD direction is:

\[
\textbf{Variationally Stable Primal-Dual Distributional Optimization for Diffusion and Flow Generative Models.}
\]

It is mathematically grounded, compatible with applied mathematics, and sufficiently connected to current generative-model research.


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
