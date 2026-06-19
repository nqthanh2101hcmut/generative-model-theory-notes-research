# Statistical Error

A typical decomposition in learning theory is

\[
\text{Total Error}
\le
\text{Optimization Error}
+
\text{Statistical Error}
+
\text{Approximation Error}
+
\text{Sampling Error}.
\]

## Optimization error

The algorithm may not find the exact empirical minimizer.

## Statistical error

The empirical objective differs from the population objective because the training set is finite.

## Approximation error

The model class, for example neural networks of finite width/depth, may not contain the true score or vector field.

## Sampling error

Even if the learned score/vector field is good, numerical sampling through an ODE/SDE solver introduces discretization error.

## Research target

For diffusion and flow models, a strong theoretical result should explicitly control all four terms.
