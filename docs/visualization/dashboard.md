
# Visualization & Dashboard Guide

The dashboard is divided into three main panels to facilitate an interactive workflow.

## 1. Parameters (Left Panel)

This section controls the inputs.
-   **Market Data Source**: Toggle between "Synthetic Data" (generated on-the-fly) and "Upload File" (CSV).
-   **Synthetic Generation**: Set the "True" parameters used to generate the target surface.
-   **Calibration Settings**:
    -   **Pricing Model**: Choose the model to fit (Heston, Merton, Bates, BS).
    -   **Advanced Settings**: Select optimization algorithm (L-BFGS-B, etc.) and max iterations.

## 2. Visualizations (Middle Panel)

### Calibration Surface
A 3D interactive plot comparing the **Market** prices (blue scatter points) vs. the **Calibrated Model** prices (orange wireframe mesh).
-   **Good Fit**: The wireframe passes directly through the blue points.
-   **Interaction**: Rotate, zoom, and hover to inspect prices at specific Strike/Maturity coordinates.

### Calibration Error (Residuals)
A 3D scatter plot of the residuals: $\text{Error} = C_{market} - C_{model}$.
-   **Analysis**: Look for patterns. Random scatter around 0 indicates good fit. Systematic bias (e.g., all OTM options priced too high) suggests model misspecification.

### Implied Volatility Smile
Plots the Implied Volatility (IV) against Strike for different maturities.
-   **Smile/Skew**: Demonstrates the model's ability to capture the "volatility smile" observed in real markets.

## 3. Reporting (Right Panel)

-   **Option Price**: Real-time status. "Waiting..." indicates idle. "Optimizing..." (with glowing indicator) shows activity.
-   **RMSE**: Root Mean Squared Error. The definitive metric of calibration success.
-   **Greeks & Metrics**: Displays the calibrated parameter set ($\Theta$) and derived Greeks.
-   **Important Equations**: Shows the mathematical formulation of the selected model for reference.
