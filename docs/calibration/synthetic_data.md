
# Synthetic Data Generation

To validate calibration algorithms without real market feeds, the platform includes a robust synthetic data generator.

## Base Model: Heston

The synthetic "market" is generated using the Heston model parameters defined in the left panel of the dashboard. This ensures the ground truth is known, allowing us to perfectly measure calibration accuracy.

### Configuration

-   **Spot ($S_0$)**: Underlying price (default 100).
-   **Rate ($r$)**: Risk-free rate (default 5%).
-   **Structure**: The generator creates a grid of:
    -   **Maturities ($T$)**: 0.5, 1.0, 2.0 years.
    -   **Moneyness ($K/S$)**: 0.8 to 1.2 (inclusive of OTM, ATM, ITM).

## Adding Noise

Real market data is never perfect. We simulate this by adding Gaussian noise to the theoretical Heston prices.

$$
C_{market} = C_{Heston}(\Theta_{true}) \times (1 + \epsilon), \quad \epsilon \sim \mathcal{N}(0, \text{noise\_level}^2)
$$

The **Noise (%)** slider controls the standard deviation of this multiplicative noise.
-   **0%**: Perfect data. Calibration should yield RMSE $\approx 0$.
-   **1-5%**: Realistic market conditions. Allows testing robustness of the optimizer.
