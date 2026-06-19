# Stability of Minimizers

Let \(F_n\) be an approximate objective and \(F\) the limiting objective.

A typical stability statement is:

If

\[
F_n \xrightarrow{\Gamma} F
\]

and \(F_n\) is equicoercive, then every cluster point of minimizers \(x_n\in\arg\min F_n\) belongs to \(\arg\min F\).

If \(F\) has a unique minimizer \(x^\star\), then

\[
x_n\to x^\star.
\]

## Application to generative model theory

Let \(F_n\) be an empirical distributional objective and \(F\) be the population objective. Stability means that the learned generative distribution does not change dramatically under finite-sample approximation.

## Possible theorem template

Let \(\mathcal{X}\) be compact and suppose \(\ell_n\to\ell\) uniformly. Define

\[
F_n(\rho)=\int \ell_n d\rho + \lambda\mathrm{KL}(\rho\|\rho_0).
\]

Then under suitable assumptions,

\[
F_n \xrightarrow{\Gamma} F
\]

and minimizers converge.
