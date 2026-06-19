# 12-Month Roadmap

This roadmap is designed for the research direction:

> Optimization, statistical and variational foundations of diffusion and flow-based generative models.

## Month 1-2: Mathematical and coding foundation

### Theory

- Real analysis: compactness, continuity, lower semicontinuity.
- Linear algebra: eigenvalues, positive definite matrices, quadratic forms.
- Probability basics: random variables, expectation, convergence in distribution.
- Python/PyTorch basics.

### Output

- `notes/probability-statistics/01_measure_probability.md`
- `projects/01_simplex_kl_optimization/simplex_kl.py`

## Month 3-4: Optimization and KKT

### Theory

- Convex sets and convex functions.
- Lagrangian and KKT conditions.
- Fenchel duality.
- Mirror descent and primal-dual algorithms.
- KL-regularized optimization.

### Output

- `notes/optimization/02_kkt_duality.md`
- `notes/optimization/04_distributional_optimization.md`
- `projects/02_primal_dual_constraints/primal_dual_simplex.py`

## Month 5-6: Probability measures and statistics

### Theory

- Probability measures.
- Empirical measures.
- Weak convergence.
- Wasserstein distance.
- Statistical error and generalization.

### Output

- Note on population vs empirical objectives.
- Experiment measuring empirical error as sample size increases.

## Month 7-8: Variational analysis and optimal transport

### Theory

- Existence of minimizers.
- Stability of minimizers.
- Gamma convergence and epi convergence.
- Kantorovich OT.
- Entropic OT and Sinkhorn.

### Output

- `notes/variational-analysis/03_stability_of_minimizers.md`
- `notes/optimal-transport/02_entropic_ot_sinkhorn.md`
- `projects/03_sinkhorn_ot/sinkhorn_demo.py`

## Month 9-10: Diffusion models and flow matching

### Theory

- Score matching.
- Denoising score matching.
- Forward/reverse SDE.
- Probability flow ODE.
- Flow matching.
- Conditional flow matching.

### Output

- `projects/04_flow_matching_2d/flow_matching_toy.py`
- `projects/05_score_matching_2d/score_matching_toy.py`

## Month 11-12: First research problem

Choose one:

1. KKT and stability of KL-regularized distributional optimization.
2. Primal-dual algorithms for constrained distributional optimization.
3. Variational convergence of minibatch OT objectives in flow matching.

### Output

- 8-10 page research proposal.
- One reproducible toy experiment.
- One theorem/proof note.
- One paper summary pack with 10-20 references.
