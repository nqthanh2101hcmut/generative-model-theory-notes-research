# Kantorovich Optimal Transport and Wasserstein Distance

## Coupling

Given probability measures \(\mu\) and \(\nu\), a coupling is a joint distribution \(\pi\) whose marginals are \(\mu\) and \(\nu\).

The set of couplings is denoted by

\[
\Pi(\mu,\nu).
\]

## Kantorovich problem

\[
\inf_{\pi\in\Pi(\mu,\nu)}
\int c(x,y)d\pi(x,y).
\]

For \(c(x,y)=\|x-y\|^p\), the p-Wasserstein distance is

\[
W_p(\mu,\nu)=
\left(
\inf_{\pi\in\Pi(\mu,\nu)}
\int \|x-y\|^p d\pi(x,y)
\right)^{1/p}.
\]

## Connection to flow matching

OT can be used to pair initial noise points with target data points, producing straighter transport paths and more stable vector field learning.
