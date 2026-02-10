
# Calibration Algorithms

The platform supports various optimization algorithms to minimize the difference between model and market prices.

## Local Optimizers

These algorithms are fast but require a good initial guess to avoid getting stuck in local minima.

-   **L-BFGS-B**: Limited-memory Broyden–Fletcher–Goldfarb–Shanno (Bounded). The default and generally most efficient choice for smooth problems with bounds (e.g., variance must be positive).
-   **SLSQP**: Sequential Least Squares Programming. Excellent for constrained problems, often converging with fewer function evaluations than L-BFGS-B.
-   **TNC**: Truncated Newton Conjugate-Gradient.

## Global Optimizers

These algorithms explore the parameter space more broadly. They are slower but more robust to initialization.

-   **Differential Evolution**: Starts with a population of candidates. Very robust for the Heston error surface, which often has multiple local minima (e.g., trade-off between $\kappa$ and $\xi$).
-   **SHGO**: Simplicial Homology Global Optimization. Good for finding the global minimum in low-dimensional bounded problems.
-   **Dual Annealing**: Generalized Simulated Annealing.

## Strategy

For best results, we recommend a **hybrid approach**:
1.  Run a global optimizer (like Differential Evolution) with loose tolerance to find a good basin of attraction.
2.  Use the result as the initial guess for a local optimizer (L-BFGS-B) to polish the solution to high precision.
