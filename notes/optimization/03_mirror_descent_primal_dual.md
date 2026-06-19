# Mirror Descent, Dual Averaging, and Primal-Dual Algorithms

## Mirror descent

Mirror descent generalizes gradient descent by replacing Euclidean distance with a Bregman divergence.

Given a strongly convex mirror map \(\psi\), the Bregman divergence is

\[
D_\psi(x,y)=\psi(x)-\psi(y)-\langle \nabla\psi(y),x-y\rangle.
\]

The mirror descent update is

\[
x_{k+1}=\arg\min_{x\in C}\left\{
\eta_k\langle g_k,x\rangle + D_\psi(x,x_k)
\right\}.
\]

## Entropic mirror descent on the simplex

On the probability simplex, choose

\[
\psi(p)=\sum_i p_i\log p_i.
\]

Then the mirror descent update becomes exponentiated gradient:

\[
p_{k+1,i}
\propto
p_{k,i}\exp(-\eta_k g_{k,i}).
\]

## Primal-dual update

For constrained problems,

\[
\min_x f(x)\quad\text{s.t.}\quad g(x)\le 0,
\]

a simple primal-dual method is

\[
x_{k+1}=\arg\min_x\{\langle \nabla_x\mathcal{L}(x_k,\lambda_k),x\rangle + \frac{1}{\eta_k}D_\psi(x,x_k)\},
\]

\[
\lambda_{k+1}=[\lambda_k+\eta_k g(x_k)]_+.
\]

## Connection to generative models

Primal-dual methods naturally appear when generative models must satisfy constraints, for example safety, diversity, fairness, cost, or physics constraints.
