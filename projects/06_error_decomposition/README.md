# Project 06: Error Decomposition Template

## Goal

Build a template for tracking four types of errors in generative model experiments:

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

## Run

```bash
python projects/06_error_decomposition/error_decomposition_template.py
```

## Output

A toy plot showing how different error terms may scale with training steps, sample size, model size, and solver steps.
