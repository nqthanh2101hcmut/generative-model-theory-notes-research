---
title: "Optimization and Variational Analysis"
subtitle: "KKT Conditions, Duality, Stability, and Distributional Optimization"
author: "Thanh Nguyen Quoc"
date: "2026"
---

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
