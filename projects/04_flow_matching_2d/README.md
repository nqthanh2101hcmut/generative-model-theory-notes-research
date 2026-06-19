# Project 04: Toy Flow Matching in 2D

## Goal

Train a small neural vector field to transport standard Gaussian noise into a 2D Gaussian mixture using a conditional flow matching objective.

The path is

\[
x_t=(1-t)x_0+t x_1,
\]

and the target velocity is

\[
u_t=x_1-x_0.
\]

## Run

```bash
python projects/04_flow_matching_2d/flow_matching_toy.py
```

## Output

- Training loss.
- Generated samples.
- Target samples.
