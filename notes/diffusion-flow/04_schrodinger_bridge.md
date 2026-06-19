# Schrödinger Bridge

The Schrödinger bridge problem asks for the most likely stochastic process connecting two endpoint distributions.

## Path-space formulation

Given a reference path measure \(\mathbb{Q}\), solve

\[
\min_{\mathbb{P}}
\mathrm{KL}(\mathbb{P}\|\mathbb{Q})
\]

subject to

\[
X_0\sim\mu_0,
\qquad
X_1\sim\mu_1.
\]

## Relation to optimal transport

Schrödinger bridge is an entropy-regularized version of dynamic optimal transport.

As entropy goes to zero, the bridge can approach optimal transport under suitable conditions.

## Relation to generative models

Schrödinger bridge connects:

- diffusion models;
- score matching;
- flow matching;
- entropic optimal transport;
- stochastic control.
