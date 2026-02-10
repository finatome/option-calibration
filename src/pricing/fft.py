import numpy as np
from scipy.fft import fft

def carr_madan_price(S0, K, T, r, char_func, alpha=1.1, N=4096, B=200):
    """
    Price Call options using Carr-Madan (1999) FFT method.
    
    Args:
        S0 (float): Spot price
        K (float or array-like): Strike price(s)
        T (float): Time to maturity
        r (float): Risk-free rate
        char_func (function): Characteristic function phi(u)
        alpha (float): Damping factor (default 1.1)
        N (int): Number of points in FFT grid (power of 2, e.g., 4096)
        B (float): Upper bound of integration domain (e.g., 200)

    Returns:
        float or np.array: Call option price(s) corresponding to K
    """
    # 1. Discretization
    dv = B / N
    du = 2 * np.pi / B  # Nyquist relation: dv * du = 2 * pi / N
    
    # 2. Construct grid in Fourier space (u)
    u = np.arange(N) * dv
    
    # 3. Construct weights for Simpson's rule (optional, here using trapezoidal/simple summation)
    # The paper uses a specific weight, but standard FFT usually implies uniform weights with adjustments.
    # We'll use the standard summation approach: sum_{j=0}^{N-1} e^{-i * 2pi * j * n / N} * x_j
    
    # Carr-Madan specific weights:
    # w = 1/3 * (3 + (-1)**(np.arange(N) + 1))  # Simpson's weights if used
    # But usually simple delta is enough if N is large. Let's use simpler weights for clarity and speed.
    weight = np.ones(N)
    weight[0] = 0.5
    weight[-1] = 0.5
    
    # 4. Compute characteristic function values
    # psi(v) = e^{-rT} * phi(v - (alpha+1)i) / (alpha^2 + alpha - v^2 + i(2alpha+1)v)
    v_j = u # u is the grid of v values in the paper
    
    # Careful with char_func arguments. It expects (u, ...). 
    # The integration variable is v. 
    # Argument to phi is (v - (alpha + 1)*1j)
    phi_val = char_func(v_j - (alpha + 1) * 1j)
    
    denominator = (alpha**2 + alpha - v_j**2 + 1j * (2 * alpha + 1) * v_j)
    psi = np.exp(-r * T) * phi_val / denominator
    
    # 5. FFT
    # x_j = e^{-i * b * v_j} * psi(v_j) * weight * dv
    # But Carr-Madan defines log-strike grid k_u = -b + lambda * u
    # Let's follow the standard FFT pricing more strictly.
    
    # Effective Upper Bound for log-strike integration
    # b = 0.5 * N * du = 0.5 * N * (2pi/B) = N*pi/B # This is range of k
    # No, let's use the layout from a standard reference.
    
    # Lambda (grid spacing for log-strike k)
    lam = 2 * np.pi / (N * dv) # = du in some notations, but here dv is integration step
    
    # Log-strikes grid
    # k_m = -b + lam * m, where b = N * lam / 2
    b = N * lam / 2
    k_grid = -b + lam * np.arange(N)
    
    # FFT Input Array
    # x_j = e^{i * b * v_j} * psi(v_j) * weight * dv   (Note the sign depends on FFT def)
    # standard fft computes sum x_j * e^{-i 2pi j m / N}
    # We want integral e^{-i v k} ...
    
    fft_input = np.exp(1j * b * v_j) * psi * weight * dv
    
    # Execute FFT
    fft_output = fft(fft_input)
    
    # Call Prices C(k)
    # C(k_m) = e^{-alpha * k_m} / pi * Re(fft_output[m])
    call_prices_grid = np.exp(-alpha * k_grid) / np.pi * np.real(fft_output)
    
    # 6. Interpolation
    # We use log-moneyness k = log(K/S0)
    # The FFT yields values C(k) which represent Bloom-Scholes like prices normalized by S0?
    # If the CF is for ln(S_T/S_0), then option payoff is (S_T/S_0 - K/S_0)^+ = 1/S_0 * (S_T - K)^+
    # So the computed price C(k) is Price / S0.
    # Therefore Price = S0 * C(k).
    
    target_k = np.log(np.atleast_1d(K) / S0)
    
    # Interpolate
    interpolated_prices_norm = np.interp(target_k, k_grid, call_prices_grid)
    
    # Scale back
    prices = S0 * interpolated_prices_norm
    
    # If single input, return scalar
    if np.ndim(K) == 0:
        return prices[0]
    return prices
