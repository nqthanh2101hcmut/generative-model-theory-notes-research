# Entropic OT and Sinkhorn Algorithm

## Discrete OT

Given weights \(a\in\Delta_n\), \(b\in\Delta_m\), and cost matrix \(C\in\mathbb{R}^{n\times m}\), the OT problem is

\[
\min_{P\ge 0}
\langle P,C\rangle
\quad\text{s.t.}\quad
P\mathbf{1}=a,
\quad
P^T\mathbf{1}=b.
\]

## Entropic regularization

\[
\min_{P\ge 0}
\langle P,C\rangle
+
\varepsilon\sum_{i,j}P_{ij}(\log P_{ij}-1)
\]

subject to the same marginal constraints.

## Sinkhorn form

The solution has the form

\[
P^\star=\operatorname{diag}(u)K\operatorname{diag}(v),
\qquad
K_{ij}=\exp(-C_{ij}/\varepsilon).
\]

Sinkhorn updates:

\[
u = a/(Kv),
\qquad
v=b/(K^Tu).
\]

## Why entropy helps

Entropy makes the OT problem smoother and computationally easier, but introduces bias relative to exact OT.
