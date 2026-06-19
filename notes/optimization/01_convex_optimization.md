# Convex Optimization

## Definition: convex set

A set \(C\subseteq\mathbb{R}^d\) is convex if for every \(x,y\in C\) and every \(\theta\in[0,1]\),

\[
\theta x + (1-\theta)y \in C.
\]

## Definition: convex function

A function \(f:C\to\mathbb{R}\) is convex if

\[
f(\theta x+(1-\theta)y)
\le
\theta f(x)+(1-\theta)f(y).
\]

## First-order condition

If \(f\) is differentiable and convex, then

\[
f(y) \ge f(x) + \langle \nabla f(x), y-x\rangle.
\]

This inequality is central in optimization and is repeatedly used to prove convergence rates.

## Basic gradient descent

For unconstrained minimization,

\[
x_{k+1}=x_k-\eta_k \nabla f(x_k).
\]

For smooth convex \(f\), the descent lemma is

\[
f(y)\le f(x)+\langle \nabla f(x),y-x\rangle + \frac{L}{2}\|y-x\|^2.
\]

## Connection to generative models

Training losses for score matching and flow matching are typically nonconvex in neural network parameters. However, many distributional formulations are convex in the probability measure \(\rho\), especially when KL regularization is included.
