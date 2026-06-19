---
title: "Probability and Statistical Foundations"
subtitle: "Measures, Empirical Distributions, Generalization, and Error Decomposition"
author: "Thanh Nguyen Quoc"
date: "2026"
---

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
