# Model Calibration

Model calibration is the process of adjusting the parameters of a stochastic model so that its outputs match observed market data as closely as possible.

## Objective Function

The core of calibration is minimizing an error metric (loss function) between market prices ($C_{market}$) and model prices ($C_{model}$):

$$
\min_{\Theta} \sum_{i=1}^N w_i \left( C_{market}^i - C_{model}^i(\Theta) \right)^2
$$

Common metrics include:
- **Mean Squared Error (MSE)**: Standard Euclidean distance.
- **Root Mean Squared Error (RMSE)**: Square root of MSE, interpreted in price units.
- **Relative Error**: Normalized by price, useful when options span a wide range of values.

## Optimization Algorithms

Finding the optimal parameters $\Theta$ is a non-linear optimization problem.

### Local Search (Gradient-Based)
- **L-BFGS-B**: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm. Efficient for smooth functions with bounds.
- **SLSQP**: Sequential Least Squares Programming. Good for constrained optimization.
- **TNC**: Truncated Newton method.

### Global Search (Derivative-Free)
- **Nelder-Mead**: Simplex algorithm. Robust to noisy functions but slower convergence.
- **Powell**: Conjugate direction method.

## Typical Parameters

| Model | Parameters to Calibrate | Description |
| :--- | :--- | :--- |
| **Black-Scholes** | $\sigma$ | Constant volatility |
| **Merton** | $\sigma, \lambda, \mu_J, \delta$ | Diffusive vol, Jump intensity, Jump mean/std |
| **Heston** | $v_0, \kappa, \theta, \xi, \rho$ | Initial vol, Mean reversion, Long-run var, Vol-vol, Correlation |
