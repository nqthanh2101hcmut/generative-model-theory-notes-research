# Measure-Theoretic Probability

## Probability space

A probability space is a triple

\[
(\Omega,\mathcal{F},\mathbb{P}),
\]

where:

- \(\Omega\) is the sample space.
- \(\mathcal{F}\) is a sigma-algebra.
- \(\mathbb{P}\) is a probability measure.

## Random variable

A random variable is a measurable map

\[
X:\Omega\to\mathcal{X}.
\]

Its distribution is the pushforward measure

\[
\mu = X_\#\mathbb{P}.
\]

This means

\[
\mu(A)=\mathbb{P}(X\in A).
\]

## Pushforward measure

If \(T:\mathcal{X}\to\mathcal{Y}\) and \(X\sim\mu\), then \(Y=T(X)\) has distribution

\[
T_\#\mu.
\]

This is the mathematical expression of a generative model: transform simple noise into data.

## Generative model view

Let \(Z\sim p_0\), where \(p_0\) is a simple noise distribution. A generator \(G_\theta\) defines

\[
X=G_\theta(Z),
\]

so the model distribution is

\[
p_\theta=(G_\theta)_\#p_0.
\]
