# Merton Jump Diffusion Model

The Merton (1976) model incorporates discontinuous jumps in the asset price process, better capturing extreme market moves and fat-tailed return distributions.

## Model Dynamics

The asset price evolution includes a standard geometric Brownian motion plus a Poisson jump process:

$$
\frac{dS_t}{S_{t-}} = (r - \lambda k)dt + \sigma dW_t + (e^J - 1)dN_t
$$

where:
- $r$: Risk-free rate.
- $\lambda$: Mean arrival rate of jumps (Poisson intensity).
- $k = E[e^J - 1]$: Expected jump size correction (martingale adjustment).
- $\sigma$: Diffusive volatility.
- $N_t$: Poisson process counting jumps.
- $J \sim N(\mu_J, \delta^2)$: Jump size (log-normal).

## Pricing

The European Call option price under Merton's model can be expressed as an infinite weighted sum of Black-Scholes prices:

$$
C_{Merton} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T} (\lambda' T)^n}{n!} C_{BS}(S_n, K, T, r_n, \sigma_n)
$$

where the parameters are adjusted for the number of jumps $n$.

Alternatively, using the Lewis (2001) Fourier transform approach, the characteristic function is:

$$
\phi(u, T) = \exp \left( i u \omega T - \frac{1}{2} u^2 \sigma^2 T + \lambda T ( e^{i u \mu_J - u^2 \delta^2/2} - 1 ) \right)
$$

## Calibration

Calibration seeks to find $\Theta = \{\sigma, \lambda, \mu_J, \delta\}$ that minimizes the root mean squared error (RMSE) between model and market prices:

$$
RMSE = \sqrt{\frac{1}{N} \sum_{i=1}^N (C_{market}^i - C_{Merton}^i(\Theta))^2}
$$
