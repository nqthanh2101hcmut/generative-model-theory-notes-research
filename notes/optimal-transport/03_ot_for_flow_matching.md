# Optimal Transport for Flow Matching

Flow matching learns a vector field \(v_t\) such that particles evolve according to

\[
\frac{dX_t}{dt}=v_t(X_t)
\]

and transform a base distribution into a data distribution.

## Linear interpolation path

Given a pair \((x_0,x_1)\), define

\[
x_t=(1-t)x_0+t x_1.
\]

The target velocity is

\[
u_t=x_1-x_0.
\]

## Random pairing vs OT pairing

Random pairing may produce crossing trajectories and complex vector fields.

OT pairing attempts to pair samples so that the total transport cost is small, often producing simpler trajectories.

## Research questions

- Does minibatch OT introduce bias?
- Does entropic OT improve stability?
- Can variational convergence justify minibatch approximations?
