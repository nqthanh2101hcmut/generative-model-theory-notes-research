# Flow Matching

Flow matching learns a time-dependent vector field

\[
v_\theta(x,t)
\]

so that particles move from noise to data through an ODE

\[
\frac{dX_t}{dt}=v_\theta(X_t,t).
\]

## Continuity equation

If \(p_t\) is the distribution of \(X_t\), then

\[
\partial_t p_t + \nabla\cdot(p_t v_t)=0.
\]

## Conditional flow matching

Given a pair \((x_0,x_1)\), define

\[
x_t=(1-t)x_0+t x_1.
\]

The target velocity is

\[
u_t=x_1-x_0.
\]

The loss is

\[
\mathbb{E}_{t,x_t}\left[\|v_\theta(x_t,t)-u_t\|^2\right].
\]

## Research direction

- Stability under vector field approximation.
- Flow trajectory straightness.
- OT pairing and minibatch bias.
- ODE discretization error.
