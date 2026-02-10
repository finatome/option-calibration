
# Bates Model (1996)

The Bates model extends the Heston Stochastic Volatility model by adding Merton-style jumps to the log-price process. This allows it to capture both the volatility smile (via stochastic vol) and short-term skews/kurtosis (via jumps), making it highly effective for calibrating across a wide range of maturities.

## Dynamics

The risk-neutral dynamics are:

$$
\begin{aligned}
\frac{dS_t}{S_{t-}} &= (r - \lambda k) dt + \sqrt{v_t} dW_t^S + (e^J - 1) dN_t \\
dv_t &= \kappa (\theta - v_t) dt + \xi \sqrt{v_t} dW_t^v
\end{aligned}
$$

Where:
- $v_t$: Instantaneous variance
- $dW_t^S, dW_t^v$: Correlated Brownian motions with correlation $\rho$
- $dN_t$: Poisson process with intensity $\lambda$ (Jump arrival rate)
- $J$: Jump size, $J \sim \mathcal{N}(\mu_J, \delta^2)$
- $k = E[e^J] - 1 = e^{\mu_J + 0.5\delta^2} - 1$: Compensator to maintain martingale property

## Parameters

The model has 8 calibration parameters:
1.  $v_0$: Initial variance
2.  $\kappa$: Rate of mean reversion
3.  $\theta$: Long-run variance
4.  $\xi$: Volatility of variance (Vol-of-Vol)
5.  $\rho$: Correlation
6.  $\lambda$: Jump intensity
7.  $\mu_J$: Mean log-jump size
8.  $\delta$: Std dev of log-jump size

## Characteristic Function

Because the jump and diffusion components are independent (conditional on the filtration), the characteristic function is the product of the Heston CF and the Merton jump CF:

$$
\phi_{Bates}(u) = \phi_{Heston}(u) \times \phi_{MertonJump}(u)
$$

This modularity allows for efficient semi-analytical pricing using FFT or integration methods (Lewis, 2001).
