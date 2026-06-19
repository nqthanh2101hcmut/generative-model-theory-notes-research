# Research Agenda

## Long-term theme

**Optimization and statistical foundations of generative models.**

The core objective is to study modern generative models through the mathematical lens of optimization on probability spaces, variational stability, statistical learning theory, and optimal transport.

## Central mathematical problem

A generic entropy-regularized distributional optimization problem is

\[
\min_{\rho \in \mathcal{P}(\mathcal{X})}
F(\rho)
:=
\mathbb{E}_{x\sim\rho}[\ell(x)]
+
\lambda \mathrm{KL}(\rho\|\rho_0)
+
\gamma \mathrm{OT}_\varepsilon(\rho,\mu),
\]

possibly under constraints

\[
\mathbb{E}_{x\sim\rho}[c_j(x)] \le b_j, \qquad j=1,\dots,m.
\]

## Research questions

### 1. Optimality and duality

- When does the problem admit a minimizer?
- What are the KKT conditions?
- When does strong duality hold?
- How can the dual variables be interpreted in generative model alignment?

### 2. Variational stability

- If empirical data \(\mu_n\) replaces the population distribution \(\mu\), do minimizers remain stable?
- If minibatch OT replaces full OT, what is the induced bias?
- Can Gamma/epi convergence be used to prove convergence of minimizers?

### 3. Algorithms

- Can mirror descent, dual averaging, or primal-dual methods solve the distributional problem?
- What convergence rate can be obtained?
- What is the constraint violation rate?

### 4. Generative model connection

- How does this framework describe diffusion alignment?
- How does entropic OT connect to flow matching and Schrödinger bridges?
- Can optimization error, statistical error, approximation error, and sampling error be decomposed?

## Suggested thesis title

Vietnamese:

> Hội tụ biến phân và điều kiện tối ưu cho bài toán tối ưu trên không gian phân phối có chính quy entropy, với ứng dụng trong mô hình sinh diffusion và flow.

English:

> Variational Convergence and Optimality Conditions for Entropy-Regularized Distributional Optimization with Applications to Diffusion and Flow Generative Models.
