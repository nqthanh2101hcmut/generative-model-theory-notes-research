# Diffusion SDE

## Forward SDE

A general forward diffusion process is

\[
dX_t=f(X_t,t)dt+g(t)dW_t.
\]

The process gradually transforms data into noise.

## Reverse-time SDE

The reverse-time process has the form

\[
dX_t=
\left[f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)\right]dt
+g(t)d\bar W_t.
\]

The key unknown is the score

\[
\nabla_x\log p_t(x).
\]

## Probability flow ODE

There is also a deterministic ODE whose marginal distributions match those of the SDE:

\[
\frac{dX_t}{dt}
=
f(X_t,t)-\frac{1}{2}g(t)^2\nabla_x\log p_t(X_t).
\]

## Error sources

- Score approximation error.
- Statistical error from finite data.
- Optimization error from training.
- Sampling error from discretizing SDE/ODE.
