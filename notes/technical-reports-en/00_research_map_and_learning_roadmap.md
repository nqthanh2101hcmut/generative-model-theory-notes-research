---
title: "Research Map and Learning Roadmap"
subtitle: "Optimization, Probability, Statistics, Optimal Transport, Diffusion Models, and Flow Matching"
author: "Thanh Nguyen Quoc"
date: "2026"
---

# Abstract

These notes define a structured research program for the mathematical foundations of modern generative models. The central view is that generative modeling is a problem of learning and controlling probability distributions. Diffusion models, score-based models, flow matching, optimal-transport flows, and Schrodinger bridges can all be interpreted as different ways of transporting a simple reference distribution into a data distribution. This report describes the mathematical areas needed for this program, the links between them, and a twelve-month learning plan leading to a first research project.

# 1. Central Perspective

A generative model is not merely a neural network that outputs samples. It induces a probability distribution. If the latent distribution is denoted by \(\rho_0\), the data distribution by \(\mu\), and the generated distribution by \(\rho_\theta\), then the goal is to construct \(\theta\) such that

\[
\rho_\theta \approx \mu.
\]

For continuous-time generative models, this approximation is often realized by a time-indexed family of probability measures \((\rho_t)_{t\in[0,1]}\), where

\[
\rho_0 \approx \text{noise}, \qquad \rho_1 \approx \text{data}.
\]

The mathematical problem is therefore to understand how probability measures evolve, how their evolution can be optimized, and how numerical and statistical errors affect the final generated distribution.

A useful research-level formulation is an optimization problem over probability measures:

\[
\min_{\rho \in \mathcal P(\mathcal X)}
F(\rho)
=
\mathbb E_{x\sim \rho}[\ell(x)]
+
\lambda \mathrm{KL}(\rho\|\rho_0)
+
\gamma \mathrm{OT}_\varepsilon(\rho,\mu),
\]

possibly subject to expectation constraints

\[
\mathbb E_{x\sim \rho}[c_j(x)] \le b_j,
\qquad j=1,\ldots,m.
\]

This formulation contains several ingredients that appear in contemporary generative modeling: distributional objectives, entropy or KL regularization, optimal transport, constrained alignment, stability of solutions, and primal-dual algorithms.

# 2. Mathematical Areas and Their Roles

The following table summarizes the main mathematical areas.

| Area | Main objects | Role in generative modeling |
|---|---|---|
| Measure-theoretic probability | probability measures, pushforwards, weak convergence | formalizes data and model distributions |
| Statistics | empirical measures, estimation error, generalization | explains learning from finite data |
| Convex and nonconvex optimization | objective functions, KKT, duality, algorithms | describes training and alignment objectives |
| Variational analysis | lower semicontinuity, Gamma convergence, stability | proves existence and stability of minimizers |
| Optimal transport | couplings, Wasserstein distances, dynamic transport | constructs paths between distributions |
| Information theory | entropy, KL, Fisher divergence | measures discrepancy and regularizes objectives |
| SDE and stochastic processes | Brownian motion, Langevin dynamics, reverse SDE | describes diffusion and score-based models |
| ODE/PDE and dynamical systems | vector fields, continuity equation, Fokker-Planck equation | describes flow matching and density evolution |
| Numerical analysis | discretization, ODE/SDE solvers, Sinkhorn iterations | controls sampling and computational error |
| Approximation theory | neural network approximation, function classes | controls representation error |

The value of this research direction is that it does not treat generative AI as a black box. It seeks mathematical guarantees for training, sampling, stability, and alignment.

# 3. Research Questions

The program is organized around five fundamental questions.

## 3.1 Existence and Optimality

Given a distributional optimization problem with entropy, KL, or OT regularization, under what assumptions does a minimizer exist? If constraints are present, can one derive Karush-Kuhn-Tucker conditions or a saddle-point characterization?

## 3.2 Variational Stability

Suppose the population objective \(F\) is replaced by an empirical or discretized objective \(F_n\). When does

\[
F_n \to F
\]

imply convergence of minimizers? In a generative-model context, this asks whether the model learned from finite data converges to the ideal population solution.

## 3.3 Algorithmic Convergence

Can primal-dual, mirror-descent, or dual-averaging algorithms solve the distributional problem with provable convergence rates? Can one bound constraint violation and primal-dual gaps?

## 3.4 Statistical and Sampling Error

A realistic error decomposition should have the form

\[
\mathrm{Total\ error}
\le
\mathrm{Optimization\ error}
+
\mathrm{Statistical\ error}
+
\mathrm{Approximation\ error}
+
\mathrm{Sampling\ error}.
\]

The central problem is to make each term mathematically precise.

## 3.5 Connection to Diffusion and Flow Models

How do score-based diffusion models, flow matching, and Schrodinger bridges fit into a unified distributional-optimization framework? Can optimization theory explain why some paths, couplings, or regularizations lead to more stable training and faster sampling?

# 4. Twelve-Month Learning Plan

## Month 1-2: Mathematical and Programming Foundations

Study real analysis, linear algebra, probability, and Python/PyTorch basics. The goal is not encyclopedic knowledge but fluency with gradients, Jacobians, compactness, convergence, expectations, and numerical experiments.

Deliverables:

- a note on probability measures and pushforward maps;
- a PyTorch implementation of gradient descent and logistic regression;
- a short mathematical note: "Why generative modeling is distributional optimization."

## Month 3-4: Convex Optimization and KKT Theory

Study convex sets, convex functions, Lagrangians, KKT conditions, Fenchel duality, mirror descent, and primal-dual algorithms.

Deliverables:

- a proof of KKT conditions for a constrained convex problem;
- a primal-dual implementation for a constrained simplex problem;
- a note on KL-regularized optimization.

## Month 5-6: Probability Measures and Statistical Learning

Study weak convergence, empirical measures, Wasserstein distance, total variation distance, concentration inequalities, statistical error, and generalization.

Deliverables:

- a note on population objectives versus empirical objectives;
- an experiment showing convergence of empirical measures for Gaussian mixtures;
- a summary of statistical error in score estimation.

## Month 7-8: Variational Analysis and Optimal Transport

Study lower semicontinuity, coercivity, Gamma convergence, stability of minimizers, Kantorovich transport, Wasserstein distances, entropic OT, and the Sinkhorn algorithm.

Deliverables:

- a proof of existence for an entropy-regularized distributional problem;
- a Sinkhorn implementation;
- an OT heatmap for toy distributions.

## Month 9-10: Diffusion and Flow Matching

Study score matching, denoising score matching, diffusion SDEs, reverse-time SDEs, probability-flow ODEs, continuity equations, and flow matching objectives.

Deliverables:

- a toy score-matching project in two dimensions;
- a toy flow-matching project in two dimensions;
- a note comparing score fields and vector fields.

## Month 11-12: First Research Problem

Choose one of the following initial problems:

1. KKT conditions and stability for KL-regularized distributional optimization.
2. Primal-dual algorithms for constrained distributional optimization.
3. Variational convergence of minibatch entropic OT objectives in flow matching.

Deliverables:

- a ten-page research proposal;
- a theorem roadmap;
- a reproducible toy experiment;
- a literature map with twenty to thirty references.

# 5. Recommended First Research Direction

A coherent first direction is:

\[
\textbf{Variationally Stable Entropy-Regularized Distributional Optimization}
\]

with applications to diffusion and flow-based generative models. This direction is mathematically grounded because it relies on variational analysis, optimization, probability, and optimal transport. It is also relevant to current generative modeling because KL regularization, entropy, optimal transport, and distributional alignment are central to diffusion alignment, flow matching, and Schrodinger bridge methods.

# 6. Expected Research Identity

A concise research identity for a CV or personal statement is:

> My research interests lie in optimization theory, variational analysis, probability, statistics, and optimal transport for the mathematical foundations of diffusion and flow-based generative models.

A more technical version is:

> I am interested in entropy-regularized distributional optimization, variational convergence, optimality conditions, primal-dual algorithms, and statistical-sampling guarantees for generative models.
