# Weak and Wasserstein Convergence

## Weak convergence

A sequence of probability measures \(\mu_n\) converges weakly to \(\mu\), written

\[
\mu_n \Rightarrow \mu,
\]

if for every bounded continuous function \(f\),

\[
\int f\,d\mu_n \to \int f\,d\mu.
\]

## Wasserstein distance

For \(p\ge 1\), the p-Wasserstein distance is

\[
W_p(\mu,\nu)
=
\left(
\inf_{\pi\in\Pi(\mu,\nu)}
\int \|x-y\|^p d\pi(x,y)
\right)^{1/p}.
\]

Here \(\Pi(\mu,\nu)\) is the set of couplings with marginals \(\mu\) and \(\nu\).

## Why Wasserstein matters in generative models

Wasserstein distance is sensitive to the geometry of samples. It can compare distributions even when their supports do not overlap, unlike KL divergence.

This makes it useful for flow matching, optimal transport, and generative model evaluation.
