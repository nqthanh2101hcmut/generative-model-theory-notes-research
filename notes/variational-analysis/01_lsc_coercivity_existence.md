# Lower Semicontinuity, Coercivity, and Existence

## Lower semicontinuity

A function \(F:X\to\mathbb{R}\cup\{+\infty\}\) is lower semicontinuous if

\[
F(x)\le \liminf_{n\to\infty}F(x_n)
\]

whenever \(x_n\to x\).

## Coercivity

A function is coercive if its sublevel sets

\[
\{x:F(x)\le c\}
\]

are compact or precompact.

## Direct method

To prove existence of a minimizer:

1. Take a minimizing sequence \(x_n\).
2. Use coercivity to extract a convergent subsequence.
3. Use lower semicontinuity to show the limit is a minimizer.

## Distributional optimization version

For probability measures, compactness may come from tightness, moment bounds, or compactness of the state space.
