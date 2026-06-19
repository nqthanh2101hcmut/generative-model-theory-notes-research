# Empirical Measures

Given samples

\[
X_1,\dots,X_n\sim \mu,
\]

the empirical measure is

\[
\hat\mu_n=\frac{1}{n}\sum_{i=1}^n\delta_{X_i}.
\]

## Population vs empirical objective

Population objective:

\[
F(\theta)=\mathbb{E}_{X\sim\mu}[\ell(\theta;X)].
\]

Empirical objective:

\[
F_n(\theta)=\frac{1}{n}\sum_{i=1}^n \ell(\theta;X_i).
\]

The statistical question is whether minimizers of \(F_n\) approximate minimizers of \(F\).

## In distributional optimization

Population problem:

\[
\min_{\rho}F(\rho;\mu).
\]

Empirical problem:

\[
\min_{\rho}F(\rho;\hat\mu_n).
\]

A central research question is:

\[
\operatorname{argmin}F(\cdot;\hat\mu_n)
\to
\operatorname{argmin}F(\cdot;\mu).
\]
