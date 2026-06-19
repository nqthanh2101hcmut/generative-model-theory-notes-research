# KKT Conditions and Duality

Consider the constrained problem

\[
\min_x f(x)
\quad \text{s.t.}\quad g_i(x)\le 0,
\quad i=1,\dots,m.
\]

The Lagrangian is

\[
\mathcal{L}(x,\lambda)=f(x)+\sum_{i=1}^m \lambda_i g_i(x),
\qquad \lambda_i\ge 0.
\]

## KKT conditions

A pair \((x^\star,\lambda^\star)\) satisfies the KKT conditions if

1. **Stationarity**

\[
0\in \partial_x \mathcal{L}(x^\star,\lambda^\star).
\]

2. **Primal feasibility**

\[
g_i(x^\star)\le 0.
\]

3. **Dual feasibility**

\[
\lambda_i^\star\ge 0.
\]

4. **Complementary slackness**

\[
\lambda_i^\star g_i(x^\star)=0.
\]

## Distributional version

For

\[
\min_{\rho\in\mathcal{P}(\mathcal{X})}
F(\rho)
\quad\text{s.t.}\quad
\mathbb{E}_\rho[c_j(x)]\le b_j,
\]

the Lagrangian is

\[
\mathcal{L}(\rho,\alpha)
=
F(\rho)+\sum_{j=1}^m \alpha_j\left(\mathbb{E}_\rho[c_j(x)]-b_j\right),
\qquad \alpha_j\ge 0.
\]

The stationarity condition becomes a variational condition over probability measures.

## Research direction

In diffusion alignment, the dual variables may represent penalties for constraint violations such as safety, diversity, or preference constraints.
