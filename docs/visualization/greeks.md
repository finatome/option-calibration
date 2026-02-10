# The Greeks

"The Greeks" refer to the sensitivities of an option's price to various underlying parameters. They are essential for risk management and hedging.

## Delta ($\Delta$)
$$
\Delta = \frac{\partial V}{\partial S}
$$
-   Measures the rate of change of the option value with respect to the underlying asset's price.
-   **Black-Scholes**: $N(d_1)$ for calls, $N(d_1) - 1$ for puts.
-   **Heston/Merton**: Often calculated numerically via finite differences.

## Gamma ($\Gamma$)
$$
\Gamma = \frac{\partial^2 V}{\partial S^2}
$$
-   Measures the rate of change of Delta with respect to the underlying price.
-   It represents the curvature of the option's value.
-   **Black-Scholes**: $\frac{N'(d_1)}{S \sigma \sqrt{T}}$.

## Theta ($\Theta$)
$$
\Theta = \frac{\partial V}{\partial t}
$$
-   Measures the sensitivity of the option price to the passage of time (Time Decay).
-   Usually negative for long option positions (options lose value as they approach expiration).

## Vega ($\nu$)
$$
\nu = \frac{\partial V}{\partial \sigma}
$$
-   Measures sensitivity to volatility.
-   Indicates how much the option price changes for a 1% change in implied volatility.
-   **Black-Scholes**: $S \sqrt{T} N'(d_1)$.

## Rho ($\rho$)
$$
\rho = \frac{\partial V}{\partial r}
$$
-   Measures sensitivity to the risk-free interest rate.
-   Typically less significant than other Greeks in low-interest environments.

## Dashboard Implementation

In our calibration dashboard, Greeks are primarily derived from the calibrated Black-Scholes model for benchmark purposes, or calculated numerically for advanced models like Heston and Merton by perturbing the inputs slightly and re-pricing.
