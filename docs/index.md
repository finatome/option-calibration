# Option Price Calibration Documentation

Welcome to the documentation for the **Option Price Calibration** platform. This tool is designed for quantitative analysts and financial engineers to visualize, test, and calibrate continuous-time option pricing models.

## Overview

The platform provides a comprehensive environment for:

1.  **Synthetic Data Generation**: Simulating option market data using advanced stochastic models (Heston).
2.  **Model Calibration**: Using optimization algorithms to fit models (Heston, Merton, Black-Scholes) to market data.
3.  **Visualization**: Analyzing calibration quality through 3D surface plots, error analysis, and implied volatility smiles.

## Key Modules

### [Pricing Models](models/heston.md)
Detailed mathematical specifications and implementation details for:
-   **Heston Stochastic Volatility Model**
-   **Merton Jump Diffusion Model**
-   **Black-Scholes Model**

### [Calibration](calibration/methodology.md)
Methodologies for parameter estimation, including:
-   Risk-neutral pricing objectives (MSE/RMSE).
-   Optimization algorithms (L-BFGS-B, SLSQP, etc.).
-   Parameter stability and uniqueness.

### [Financial Concepts](visualization/greeks.md)
Background on fundamental concepts:
-   Put-Call Parity
-   The Greeks (Delta, Gamma, Vega, Theta, Rho)
-   Delta Hedging

## Getting Started

To run the application locally:

```bash
pip install -r requirements.txt
python app.py
```

Navigate to `http://localhost:8050` to access the dashboard.
