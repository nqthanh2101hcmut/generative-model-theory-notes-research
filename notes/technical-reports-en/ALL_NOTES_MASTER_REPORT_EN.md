---
title: "Generative Model Theory Notes"
subtitle: "Technical Report and Research Notes on Optimization, Probability, Statistics, Optimal Transport, Diffusion Models, and Flow Matching"
author: "Thanh Nguyen Quoc"
date: "2026"
geometry: margin=1in
fontsize: 11pt
---

# Preface

This master technical report collects a set of English research notes for the repository `generative-model-theory-notes`. The notes are written as a working academic document rather than as informal study notes. Their purpose is to support a research trajectory in applied mathematics and machine learning theory, especially the mathematical foundations of diffusion and flow-based generative models.

The guiding principle is:

\[
\textbf{Generative modeling is distributional optimization.}
\]

The report is organized into seven parts: a research map, optimization and variational analysis, probability and statistics, optimal transport and entropy, diffusion models, flow matching, and a PhD-oriented research plan.

\newpage


# Part I. Research Map and Learning Roadmap

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

\newpage


# Part II. Optimization and Variational Analysis

# Abstract

This report develops the optimization and variational-analysis foundation for a research program on generative model theory. The main goal is to understand distributional objectives with entropy, KL, and optimal-transport regularization. We review constrained optimization, KKT conditions, Lagrangian duality, mirror-descent ideas, and variational stability of minimizers. These tools provide a mathematical language for diffusion alignment, flow matching, and constrained generative modeling.

# 1. From Parameter Optimization to Distributional Optimization

Standard deep learning is often written as a parameter optimization problem:

\[
\min_{\theta\in\Theta} \mathcal L(\theta).
\]

For generative models, however, the parameter \(\theta\) is important primarily because it induces a probability distribution \(\rho_\theta\). This suggests an alternative viewpoint:

\[
\min_{\rho\in\mathcal P(\mathcal X)} F(\rho),
\]

where \(F\) measures data fidelity, regularity, alignment reward, or constraint satisfaction. This viewpoint is useful for theoretical analysis because the target object is the generated distribution itself.

A prototypical problem is

\[
\min_{\rho\in\mathcal P(\mathcal X)}
F(\rho)
=
\mathbb E_{\rho}[\ell]
+
\lambda \mathrm{KL}(\rho\|\rho_0)
+
\gamma \mathrm{OT}_\varepsilon(\rho,\mu).
\]

Here \(\rho_0\) is a reference distribution, \(\mu\) is the data distribution, \(\ell\) is a cost or negative reward, and \(\mathrm{OT}_\varepsilon\) is an entropy-regularized optimal transport cost.

# 2. Constrained Distributional Optimization

Many generative-model objectives include constraints. For instance, one may require generated samples to satisfy safety, diversity, energy, fairness, or physical constraints in expectation:

\[
\mathbb E_{\rho}[c_j(x)] \le b_j,
\qquad j=1,\ldots,m.
\]

The constrained problem is

\[
\begin{aligned}
\min_{\rho\in\mathcal P(\mathcal X)} \quad & F(\rho) \\
\text{subject to} \quad & C_j(\rho):=\mathbb E_{\rho}[c_j]-b_j \le 0,
\quad j=1,\ldots,m.
\end{aligned}
\]

The Lagrangian is

\[
\mathcal L(\rho,\alpha)
=
F(\rho)+\sum_{j=1}^m \alpha_j C_j(\rho),
\qquad \alpha_j\ge 0.
\]

A saddle-point formulation is

\[
\inf_{\rho\in\mathcal P(\mathcal X)}\sup_{\alpha\ge 0}
\mathcal L(\rho,\alpha).
\]

This formulation is central for primal-dual algorithms and for KKT-type optimality conditions.

# 3. KKT Conditions in a Finite-Dimensional Approximation

To build intuition, consider a finite sample space \(\mathcal X=\{x_1,\ldots,x_N\}\). A probability distribution is a vector \(p\in\Delta_N\), where

\[
\Delta_N=\{p\in\mathbb R^N: p_i\ge 0, \sum_{i=1}^N p_i=1\}.
\]

Consider

\[
\min_{p\in\Delta_N}
\sum_{i=1}^N p_i \ell_i
+
\lambda \sum_{i=1}^N p_i \log\frac{p_i}{q_i}
\]

subject to

\[
\sum_{i=1}^N p_i c_{j,i}\le b_j,
\qquad j=1,\ldots,m.
\]

The Lagrangian is

\[
\mathcal L(p,\alpha,\tau)
=
\sum_i p_i \ell_i
+
\lambda\sum_i p_i\log\frac{p_i}{q_i}
+
\sum_j \alpha_j\left(\sum_i p_i c_{j,i}-b_j\right)
+
\tau\left(\sum_i p_i-1\right).
\]

For an interior solution \(p_i>0\), stationarity gives

\[
\ell_i+
\lambda\left(1+\log\frac{p_i}{q_i}\right)
+
\sum_j \alpha_j c_{j,i}
+
\tau=0.
\]

Solving for \(p_i\) yields the Gibbs form

\[
p_i
=
\frac{q_i\exp\left[-\lambda^{-1}\left(\ell_i+
\sum_j \alpha_j c_{j,i}\right)\right]}
{Z(\alpha)},
\]

where \(Z(\alpha)\) is the normalizing constant. This formula shows how KL regularization produces a smooth reweighting of the reference distribution.

# 4. Duality and Interpretation

The dual variables \(\alpha_j\) can be interpreted as prices for constraint violation. If a constraint is inactive, the corresponding multiplier is zero. If a constraint is active, the multiplier is generally positive.

The complementary slackness condition is

\[
\alpha_j^\star C_j(\rho^\star)=0,
\qquad j=1,\ldots,m.
\]

This condition is highly relevant for constrained generative modeling. For example, if a generated distribution violates a diversity or safety constraint, the dual variable increases and penalizes the next primal update.

# 5. Mirror Descent and KL Geometry

Euclidean gradient descent is not always natural for probability distributions because the simplex has positivity and normalization constraints. A more suitable method is mirror descent with KL divergence:

\[
p_{k+1}
=
\arg\min_{p\in\Delta_N}
\left\{
\langle g_k,p\rangle
+
\frac{1}{\eta_k}\mathrm{KL}(p\|p_k)
\right\}.
\]

This update has the closed form

\[
p_{k+1,i}
\propto
p_{k,i}\exp(-\eta_k g_{k,i}).
\]

This multiplicative update is stable on the simplex and connects naturally with entropy-regularized distributional optimization.

# 6. Primal-Dual Algorithmic Template

For constrained distributional optimization, a basic primal-dual template is

\[
\rho_{k+1}
=
\arg\min_{\rho}
\left\{
\langle G_k,\rho\rangle
+
\frac{1}{\eta_k}\mathrm{KL}(\rho\|\rho_k)
+
\gamma\mathrm{OT}_\varepsilon(\rho,\mu_n)
\right\},
\]

\[
\alpha_{k+1}
=
\left[\alpha_k+\beta_k C(\rho_{k+1})\right]_+.
\]

The research task is to establish conditions under which

\[
\frac{1}{K}\sum_{k=1}^K
\left( F(\rho_k)-F(\rho^\star)\right)
\to 0,
\]

and

\[
\left[ C_j(\bar\rho_K)\right]_+ \to 0.
\]

# 7. Variational Stability of Minimizers

Let \(F_n\) be an empirical or approximate objective and \(F\) be the population objective. The key question is:

\[
F_n \to F
\quad \Longrightarrow \quad
\operatorname{argmin}F_n \to \operatorname{argmin}F?
\]

A common strategy is to prove:

1. compactness or tightness of near-minimizers;
2. a liminf inequality;
3. existence of recovery sequences;
4. equicoercivity or tightness of the objective family.

In Gamma-convergence terminology, one proves that \(F_n\) Gamma-converges to \(F\). If the minimizers are precompact and the limit problem has a unique minimizer, then minimizers of \(F_n\) converge to the minimizer of \(F\).

# 8. Research-Level Proposition Template

A typical theorem for this research direction may have the following form.

**Proposition.** Let \(\mathcal X\) be compact, let \(\ell_n\to \ell\) uniformly, and let \(\rho_0\) have full support. Define

\[
F_n(\rho)=\mathbb E_\rho[\ell_n]+\lambda\mathrm{KL}(\rho\|\rho_0),
\qquad
F(\rho)=\mathbb E_\rho[\ell]+\lambda\mathrm{KL}(\rho\|\rho_0).
\]

If \(\lambda>0\), then \(F_n\) converges to \(F\) uniformly on KL-bounded subsets. Under uniqueness of the minimizer of \(F\), every sequence of minimizers \(\rho_n^\star\in\operatorname{argmin}F_n\) converges to \(\rho^\star\) in the topology induced by the chosen divergence or weak convergence.

This proposition is only a template. A complete proof requires specifying topology, compactness, and continuity assumptions.

# 9. Implications for Generative Models

Optimization and variational analysis clarify several issues in generative modeling:

- KL regularization stabilizes distributional alignment.
- KKT multipliers quantify constraint trade-offs.
- Mirror descent gives natural algorithms on probability spaces.
- Variational convergence explains why empirical or minibatch objectives may approximate population objectives.
- Stability theory explains robustness under data perturbation, noisy rewards, and discretization.

# 10. Exercises and Projects

1. Prove the Gibbs-form solution for a KL-regularized linear objective on the simplex.
2. Derive KKT conditions for a constrained entropy-regularized problem.
3. Implement mirror descent on the simplex.
4. Implement a primal-dual constrained distribution optimization algorithm.
5. Experimentally compare Euclidean projected gradient and KL mirror descent.
6. Write a note explaining how dual variables can be used for constrained generative alignment.

\newpage


# Part III. Probability and Statistical Foundations

# Abstract

This report reviews the probability and statistics needed for theoretical work on generative models. Generative modeling is distribution learning: the object of interest is a probability measure rather than a single prediction. Therefore, probability measures, weak convergence, empirical distributions, Wasserstein distances, and statistical error are central. We emphasize the distinction between population objectives and empirical objectives and explain how statistical learning theory enters diffusion and flow-based generative modeling.

# 1. Probability Measures as Objects of Learning

Let \(\mathcal X\subseteq\mathbb R^d\) be the data space. A data-generating process is modeled as a probability measure \(\mu\in\mathcal P(\mathcal X)\). A generative model produces a distribution \(\rho_\theta\in\mathcal P(\mathcal X)\). The ideal learning problem is

\[
\rho_\theta \approx \mu.
\]

The approximation may be measured by total variation, Wasserstein distance, maximum mean discrepancy, KL divergence, Fisher divergence, or another statistical distance.

# 2. Pushforward Measures

If \(Z\sim \rho_0\) and \(T:\mathcal Z\to\mathcal X\) is a measurable map, then the generated variable \(X=T(Z)\) has distribution

\[
T_{\sharp}\rho_0(A)=\rho_0(T^{-1}(A))
\]

for measurable sets \(A\subseteq\mathcal X\). This is called the pushforward of \(\rho_0\) by \(T\).

Normalizing flows are based directly on pushforward maps. Continuous flows use a time-dependent vector field to define a family of maps \(T_t\), and the generated distribution is \((T_1)_{\sharp}\rho_0\).

# 3. Empirical Measures

In practice, one does not know \(\mu\). One observes samples

\[
X_1,\ldots,X_n\sim \mu.
\]

The empirical distribution is

\[
\mu_n=\frac{1}{n}\sum_{i=1}^n \delta_{X_i}.
\]

A fundamental statistical question is whether \(\mu_n\) converges to \(\mu\), and in what sense. In many settings,

\[
\mu_n \Rightarrow \mu
\]

weakly as \(n\to\infty\). However, rates depend on dimension, smoothness, and the metric used.

# 4. Population and Empirical Objectives

A population objective may be written as

\[
F(\theta)=\mathbb E_{X\sim\mu}[\ell_\theta(X)].
\]

The empirical objective is

\[
F_n(\theta)=\frac1n\sum_{i=1}^n \ell_\theta(X_i).
\]

In distributional optimization, a population objective may be

\[
F(\rho)=D(\rho,\mu)+R(\rho),
\]

where \(D\) is a discrepancy and \(R\) is a regularizer. The empirical version is

\[
F_n(\rho)=D(\rho,\mu_n)+R(\rho).
\]

The stability question is whether minimizers of \(F_n\) approximate minimizers of \(F\).

# 5. Distances Between Probability Measures

## 5.1 Total Variation

The total variation distance is

\[
\|\rho-\mu\|_{\mathrm{TV}}
=
\sup_A |\rho(A)-\mu(A)|.
\]

It is strong, but may be too strict in high-dimensional continuous settings.

## 5.2 Wasserstein Distance

For \(p\ge 1\), the \(p\)-Wasserstein distance is

\[
W_p(\rho,\mu)
=
\left(
\inf_{\pi\in\Pi(\rho,\mu)}
\int \|x-y\|^p\,d\pi(x,y)
\right)^{1/p}.
\]

Wasserstein distance is geometrically meaningful and central in optimal transport, flow matching, and generative modeling.

## 5.3 KL Divergence

If \(\rho\ll\mu\), the KL divergence is

\[
\mathrm{KL}(\rho\|\mu)
=
\int \log\left(\frac{d\rho}{d\mu}\right)d\rho.
\]

KL divergence is not symmetric and can be infinite when support conditions fail. It is nevertheless essential for entropy regularization, variational inference, and Schrodinger bridge problems.

## 5.4 Fisher Divergence

If \(\rho\) and \(\mu\) have densities, the Fisher divergence is often written as

\[
\mathcal J(\rho\|\mu)
=
\mathbb E_{X\sim \rho}
\left[
\|\nabla \log \rho(X)-\nabla \log \mu(X)\|^2
\right].
\]

Score matching is based on estimating score functions rather than normalized densities.

# 6. Statistical Error in Generative Models

A typical statistical-learning analysis decomposes the final error as

\[
\mathrm{Error}
\le
\mathrm{Approximation\ error}
+
\mathrm{Estimation\ error}
+
\mathrm{Optimization\ error}.
\]

For generative models, one often adds sampling and discretization errors:

\[
\mathrm{Total\ error}
\le
\mathrm{Approximation}
+
\mathrm{Statistical}
+
\mathrm{Optimization}
+
\mathrm{Sampling}
+
\mathrm{Discretization}.
\]

Each term has a different mathematical source.

- Approximation error: the neural network class may not represent the true score or vector field.
- Statistical error: only finitely many samples are available.
- Optimization error: training may not reach the empirical minimizer.
- Sampling error: reverse SDE or ODE sampling introduces numerical error.
- Discretization error: time discretization changes the continuous process.

# 7. Score Estimation as a Statistical Problem

In score-based diffusion models, one learns a function \(s_\theta(x,t)\) approximating

\[
\nabla_x \log p_t(x).
\]

The population score-matching objective has the form

\[
\mathcal L(\theta)=
\mathbb E_{t,X_t}
\left[
\|s_\theta(X_t,t)-\nabla_x\log p_t(X_t)\|^2
\right].
\]

The empirical version replaces expectations by finite-sample averages. The statistical question is whether small empirical score error implies small population score error and, ultimately, small distributional error in the generated samples.

# 8. Generalization for Vector Fields in Flow Matching

Flow matching learns a vector field \(v_\theta(x,t)\). A population objective may be

\[
\mathcal L(\theta)
=
\mathbb E_{t,X_t}
\left[
\|v_\theta(X_t,t)-u_t(X_t)\|^2
\right].
\]

The empirical objective is computed from sampled pairs or conditional paths. The theoretical problem is to show that generalization of the vector-field loss implies stability of the induced ODE flow and convergence of terminal distributions.

# 9. Research Directions

Important open directions include:

1. finite-sample stability of entropy-regularized distributional optimization;
2. generalization bounds for score matching under non-smooth data distributions;
3. statistical error bounds for flow matching with minibatch OT couplings;
4. convergence rates depending on intrinsic dimension rather than ambient dimension;
5. error decomposition for diffusion alignment with constrained objectives.

# 10. Exercises and Projects

1. Prove weak convergence of empirical measures under simple assumptions.
2. Compute Wasserstein distance between one-dimensional empirical distributions.
3. Simulate empirical convergence for Gaussian mixtures.
4. Estimate MMD between generated and target samples.
5. Write a report comparing TV, KL, Wasserstein, Fisher divergence, and MMD.
6. Derive an empirical-population decomposition for a simple score-matching loss.

\newpage


# Part IV. Optimal Transport, Entropy, and Schrodinger Bridges

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

\newpage


# Part V. Diffusion and Score-Based Generative Models

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

\newpage


# Part VI. Flow Matching and Continuous Flows

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

\newpage


# Part VII. Research Program and Paper Plan

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

\newpage


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
