# Heston Stochastic Volatility Model

The Heston (1993) model extends the Black-Scholes framework by allowing volatility to be a stochastic process. This captures the "volatility smile" and "skew" observed in option markets.

## Model Dynamics

The asset price $S_t$ and its variance $v_t$ follow the stochastic differential equations (SDEs):

$$
\begin{aligned}
dS_t &= r S_t dt + \sqrt{v_t} S_t dW_t^S \\
dv_t &= \kappa(\theta - v_t)dt + \xi \sqrt{v_t} dW_t^v
\end{aligned}
$$

where:
- $r$: Risk-free rate.
- $\kappa$: Mean reversion speed of volatility.
- $\theta$: Long-run average variance.
- $\xi$: Volatility of volatility (vol-vol).
- $\rho$: Correlation between asset price and volatility Brownian motions ($dW_t^S dW_t^v = \rho dt$).

## Pricing via Fourier Transform

We use the Lewis (2001) approach, which expresses the call option price as an integral involving the characteristic function $\phi(u)$ of the log-price.

### Characteristic Function

$$
\phi(u, t) = \exp(A(u,t) + B(u,t)v_0 + i u ( \ln S_0 + rt ))
$$

where $A(u,t)$ and $B(u,t)$ are solutions to Riccati equations derived from the Heston PDE.

### Pricing Formula

The price of a European Call option is given by:

$$
C_0 = S_0 - \frac{K e^{-rT}}{\pi} \int_{0}^{\infty} \text{Re}\left( \frac{e^{-i u k} \phi(u - i/2)}{u^2 + 1/4} \right) du
$$

where $k = \ln(S_0/K)$.

## Calibration

Calibration involves finding the parameters $\Theta = \{v_0, \kappa, \theta, \xi, \rho\}$ that minimize the error between model prices and observed market prices:

$$
\min_{\Theta} \sum_{i=1}^N (C_{market}^i - C_{Heston}^i(\Theta))^2
$$
