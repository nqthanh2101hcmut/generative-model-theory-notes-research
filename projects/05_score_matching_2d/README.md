# Project 05: Toy Denoising Score Matching in 2D

## Goal

Train a score network using denoising score matching on a 2D Gaussian mixture.

Given clean data \(x\) and noisy observation

\[
\tilde x = x + \sigma\epsilon,
\]

the conditional score target is

\[
-\frac{\tilde x-x}{\sigma^2}.
\]

## Run

```bash
python projects/05_score_matching_2d/score_matching_toy.py
```

## Output

- Training loss.
- Langevin samples.
- Target samples.
