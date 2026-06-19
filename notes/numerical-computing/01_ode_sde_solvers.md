# ODE and SDE Solvers

## Euler method for ODE

For

\[
\frac{dX_t}{dt}=v(X_t,t),
\]

the Euler update is

\[
X_{k+1}=X_k+h v(X_k,t_k).
\]

## Euler-Maruyama for SDE

For

\[
dX_t=f(X_t,t)dt+g(t)dW_t,
\]

the Euler-Maruyama update is

\[
X_{k+1}=X_k+h f(X_k,t_k)+g(t_k)\sqrt{h}\xi_k,
\qquad \xi_k\sim\mathcal{N}(0,I).
\]

## Sampling error

The discretization step size \(h\) introduces numerical sampling error. In generative models, this error affects the final generated distribution.
