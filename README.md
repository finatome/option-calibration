# Option Price Calibration

A professional quantitative finance dashboard for calibrating continuous-time option pricing models to market data. This tool allows users to generate synthetic market data, test calibration algorithms, and visualize the results.

## Features

- **Synthetic Data Generation**: Create realistic option market surfaces using the Heston model with adjustable parameters (Spot, Rate, Kappa, Theta, VolVol, Rho, Noise).
- **Model Calibration**: Calibrate three major pricing models to the generated data:
    - **Heston Stochastic Volatility Model** (Fourier-based pricing)
    - **Merton Jump Diffusion Model**
    - **Black-Scholes Model**
- **Advanced Optimization**:
    - Select from multiple optimization algorithms (L-BFGS-B, SLSQP, Nelder-Mead, etc.).
    - Provide custom initial guesses for model parameters.
- **Interactive Visualization**:
    - **Calibration Surface**: 3D overlay of Model vs Market prices.
    - **Error Analysis**: 3D residuals plot.
    - **Implied Volatility Smile**: Comparison of Market and Model IVs.
- **Reporting**: Real-time display of calibrated parameters, MSE/RMSE error, and status.

## Project Structure

```
├── app.py                 # Main application entry point
├── src/
│   ├── calibration/       # Optimization logic
│   ├── data/              # Synthetic data generation
│   ├── dashboard/         # Dash layout and callbacks
│   └── pricing/           # Pricing models (Heston, Merton, BS)
├── tests/                 # Integration and unit tests
└── notebooks/             # Research and reference notebooks
```

## Installation

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  Run the application:
    ```bash
    python app.py
    ```
2.  Open your browser to `http://127.0.0.1:8050/`.
3.  **Generate Data**: Use the left panel to set parameters and click "Generate Market Data".
4.  **Calibrate**: Select a model and click "Calibrate Model".
5.  **Analyze**: View the results in the middle (plots) and right (metrics) panels.

## Testing

Run the integration tests to verify the calibration workflow:
```bash
python -m unittest tests/test_calibration_workflow.py
```
