
# Black-Scholes-Merton Model

The Black-Scholes-Merton (BSM) model is the standard benchmark for option pricing. It assumes the underlying asset follows a Geometric Brownian Motion (GBM).

## Dynamics

The dynamics of the asset price $S_t$ under the risk-neutral measure Q are given by the Stochastic Differential Equation (SDE):

$$
dS_t = r S_t dt + \sigma S_t dW_t
$$

Where:
- $r$: Risk-free interest rate
- $\sigma$: Constant volatility
- $dW_t$: Wiener process (Brownian motion)

## Closed-Form Solution

For a European Call option with strike $K$ and maturity $T$, the price is given by:

$$
C(S, t) = S_t N(d_1) - K e^{-r(T-t)} N(d_2)
$$

Where $N(\cdot)$ is the cumulative distribution function of the standard normal distribution, and:

$$
d_1 = \frac{\ln(S_t/K) + (r + \frac{1}{2}\sigma^2)(T-t)}{\sigma \sqrt{T-t}}
$$

$$
d_2 = d_1 - \sigma \sqrt{T-t}
$$

For a Put option, using Put-Call Parity:
$$
P(S, t) = K e^{-r(T-t)} N(-d_2) - S_t N(-d_1)
$$
