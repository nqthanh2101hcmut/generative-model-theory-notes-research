# PyTorch Training Loop Template

A minimal PyTorch training loop:

```python
for step in range(num_steps):
    batch = sample_batch(batch_size)
    loss = compute_loss(model, batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Best practices

- Set random seeds.
- Log loss every fixed number of steps.
- Save figures to `outputs/`.
- Keep models small for toy experiments.
- Start with 2D distributions before image models.

## For theory-oriented generative modeling

Focus on experiments that test mathematical claims:

- Does KL regularization stabilize optimization?
- Does OT pairing make vector fields simpler?
- Does increasing sample size reduce empirical error?
- Does reducing ODE/SDE step size improve distribution quality?
