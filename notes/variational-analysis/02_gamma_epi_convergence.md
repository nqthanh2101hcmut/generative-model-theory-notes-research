# Gamma and Epi Convergence

## Gamma convergence

A sequence \(F_n:X\to\mathbb{R}\cup\{+\infty\}\) Gamma-converges to \(F\) if:

### Liminf inequality

For every \(x_n\to x\),

\[
F(x)\le \liminf_{n\to\infty}F_n(x_n).
\]

### Recovery sequence

For every \(x\in X\), there exists \(x_n\to x\) such that

\[
F(x)\ge \limsup_{n\to\infty}F_n(x_n).
\]

## Why it matters

If \(F_n\) Gamma-converges to \(F\) and the sequence is equicoercive, then minimizers of \(F_n\) converge to minimizers of \(F\).

## Research connection

This is useful for proving stability of:

- empirical objectives;
- discretized objectives;
- minibatch OT objectives;
- entropy-regularized approximations;
- neural approximations of score/vector fields.
