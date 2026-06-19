# Project 01: Simplex KL Optimization

## Goal

Solve the entropy/KL-regularized finite distributional optimization problem

\[
\min_{p\in\Delta_d}
\langle p,c\rangle + \lambda \mathrm{KL}(p\|q).
\]

The closed-form solution is

\[
p_i^\star \propto q_i\exp(-c_i/\lambda).
\]

## Run

```bash
python projects/01_simplex_kl_optimization/simplex_kl.py
```

## Expected output

- Prints the reference distribution, cost vector, and optimal distribution.
- Saves a plot to `outputs/simplex_kl_solution.png`.
