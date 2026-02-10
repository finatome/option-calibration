
# FFT Pricing Methodology

For models with known characteristic functions (Heston, Bates, Merton), Fourier Transform methods provide a highly efficient way to price options across a range of strikes simultaneously.

## Lewis (2001) Integration

For a single option price, we use the approach by Lewis (2001). The call price is given by:

$$
C(S_0, K, T) = S_0 - \frac{K e^{-rT}}{\pi} \int_{0}^{\infty} \text{Re} \left[ \frac{e^{-i u k} \phi(u - i/2)}{u^2 + 1/4} \right] du
$$

Where $k = \ln(S_0/K e^{-rT})$ is the log-moneyness.

This integral is computed using adaptive quadrature (Python's `scipy.integrate.quad`).

## Carr-Madan (1999) FFT

For calibrating an entire surface, evaluating the integral for every strike is slow. The Carr-Madan algorithm allows us to price options for $N$ strikes in $O(N \log N)$ time using the Fast Fourier Transform (FFT).

### Algorithm Steps

1.  **Dampening**: Define a dampened call price $c_T(k) = e^{\alpha k} C_T(k)$ to ensure square-integrability. A common choice is $\alpha = 1.5$.
2.  **Fourier Transform**: The Fourier transform of $c_T(k)$ is derived analytically from the model's characteristic function $\phi(u)$.
3.  **Discretization**: The integration domain is discretized into a grid of size $N$ (e.g., 4096) with spacing $\eta$.
4.  **FFT Application**: Apply the Discrete Fourier Transform (DFT) via FFT to compute the integral summation efficiently.
5.  **Interpolation**: The FFT outputs prices on a regular grid of strikes. We use spline interpolation to get prices for the specific market strikes.

This method effectively reduces the complexity of calibrating to large surfaces (e.g., 100+ instruments) from minutes to seconds.
