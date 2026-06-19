# Research Problem Bank

## Problem 1: KKT conditions for entropy-regularized distributional optimization

Study

\[
\min_{\rho \in \mathcal{P}(\mathcal{X})}
\mathbb{E}_{\rho}[\ell(x)] + \lambda \mathrm{KL}(\rho\|\rho_0)
\]

under moment constraints

\[
\mathbb{E}_{\rho}[c_j(x)] \le b_j.
\]

Questions:

- Does a minimizer exist?
- Is the minimizer unique?
- What are the KKT conditions?
- What is the Gibbs form of the solution?

## Problem 2: Stability of minimizers under empirical approximation

Let \(F_n\) be an empirical version of \(F\). Study whether

\[
F_n \xrightarrow{\Gamma} F
\]

implies convergence of minimizers.

Questions:

- What topology should be used: weak, Wasserstein, total variation?
- Which assumptions guarantee equicoercivity?
- Can one obtain explicit rates?

## Problem 3: Primal-dual methods for constrained generative alignment

Study a primal-dual algorithm for

\[
\min_\rho F(\rho) \quad \text{s.t.}\quad \mathbb{E}_{\rho}[c_j(x)] \le b_j.
\]

Questions:

- What is the primal-dual gap?
- How fast does constraint violation vanish?
- How does KL regularization improve stability?

## Problem 4: Variational convergence of minibatch entropic OT

In flow matching, full OT is often replaced by minibatch OT. Study:

\[
\mathrm{OT}_{\varepsilon}^{(m)}(\mu_n,\nu_n) \to \mathrm{OT}_{\varepsilon}(\mu,\nu).
\]

Questions:

- Is minibatch OT biased?
- Can Gamma convergence handle the approximation?
- How does entropy \(\varepsilon\) affect stability?

## Problem 5: Error decomposition for diffusion and flow models

Target decomposition:

\[
\mathrm{Error}
\le
\mathrm{Optimization}
+
\mathrm{Statistical}
+
\mathrm{Approximation}
+
\mathrm{Sampling}.
\]

Questions:

- Which metric is most natural: KL, TV, Wasserstein, MMD?
- How does score/vector-field error propagate to distribution error?
- How does ODE/SDE discretization affect the final samples?
