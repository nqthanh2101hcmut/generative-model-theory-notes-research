# Project 02: Primal-Dual Constraints on the Simplex

## Goal

Solve a constrained distributional optimization problem

\[
\min_{p\in\Delta_d}
\langle p,c\rangle + \lambda \mathrm{KL}(p\|q)
\quad\text{s.t.}\quad
\langle p,a\rangle \le b.
\]

The script uses an exponentiated-gradient primal update and a projected dual ascent update.

## Run

```bash
python projects/02_primal_dual_constraints/primal_dual_simplex.py
```

## Output

- Objective curve.
- Constraint violation curve.
- Final probability distribution.
