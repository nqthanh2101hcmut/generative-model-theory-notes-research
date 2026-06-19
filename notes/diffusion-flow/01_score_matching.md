# Score Matching

## Score function

For a density \(p(x)\), the score function is

\[
\nabla_x\log p(x).
\]

Score-based generative models learn an approximation

\[
s_\theta(x,t)\approx \nabla_x\log p_t(x).
\]

## Denoising score matching

Let

\[
\tilde x = x + \sigma \varepsilon,
\qquad \varepsilon\sim\mathcal{N}(0,I).
\]

The conditional score is

\[
\nabla_{\tilde x}\log p(\tilde x|x)
= -\frac{\tilde x-x}{\sigma^2}.
\]

A denoising score matching objective is

\[
\mathbb{E}\left[
\left\|s_\theta(\tilde x,\sigma)+\frac{\tilde x-x}{\sigma^2}\right\|^2
\right].
\]

## Research connection

Score approximation error propagates through reverse SDE sampling and affects the final generated distribution.
