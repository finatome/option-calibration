import numpy as np
from scipy.integrate import quad

def heston_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """
    Valuation of European call option in H93 model via Lewis (2001)
    Fourier-based approach: characteristic function.
    """
    c1 = kappa_v * theta_v
    c2 = -np.sqrt(
        (rho * sigma_v * u * 1j - kappa_v) ** 2 - sigma_v**2 * (-u * 1j - u**2)
    )
    c3 = (kappa_v - rho * sigma_v * u * 1j + c2) / (
        kappa_v - rho * sigma_v * u * 1j - c2
    )
    H1 = r * u * 1j * T + (c1 / sigma_v**2) * (
        (kappa_v - rho * sigma_v * u * 1j + c2) * T
        - 2 * np.log((1 - c3 * np.exp(c2 * T)) / (1 - c3))
    )
    H2 = (
        (kappa_v - rho * sigma_v * u * 1j + c2)
        / sigma_v**2
        * ((1 - np.exp(c2 * T)) / (1 - c3 * np.exp(c2 * T)))
    )
    char_func_value = np.exp(H1 + H2 * v0)
    return char_func_value

def heston_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0):
    """
    Fourier-based approach for Lewis (2001): Integration function.
    """
    char_func_value = heston_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0
    )
    int_func_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value

def heston_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type='C'):
    """
    Valuation of European call option in H93 model via Lewis (2001)
    
    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free rate
    - v0: Initial variance
    - kappa: Mean reversion speed of variance
    - theta: Long-term mean variance
    - xi: Volatility of volatility (sigma_v)
    - rho: Correlation between asset and variance
    - option_type: 'C' or 'P'
    """
    # Map parameters to notebook variable names for clarity
    sigma_v = xi
    kappa_v = kappa
    theta_v = theta
    
    int_value = quad(
        lambda u: heston_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0),
        0,
        np.inf,
        limit=250,
    )[0]
    
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    
    if option_type.upper().startswith('P'):
        # Put-Call Parity
        put_value = call_value - S0 + K * np.exp(-r * T)
        return put_value
        
    return call_value


def heston_mc(S0, K, T, r, v0, kappa, theta, xi, rho, option_type='C', num_paths=100, num_steps=100):
    """
    Monte Carlo Simulation for Heston Model.
    """
    dt = T / num_steps
    
    # Generate Correlated Brownian Motions
    Z1 = np.random.normal(0, 1, (num_paths, num_steps))
    Z2_uncorr = np.random.normal(0, 1, (num_paths, num_steps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2_uncorr
    
    # Initialize arrays
    S = np.zeros((num_paths, num_steps + 1))
    v = np.zeros((num_paths, num_steps + 1))
    
    S[:, 0] = S0
    v[:, 0] = v0
    
    # Simulation Loop
    # We need a loop because variance depends on previous step non-linearly (sqrt(v)) 
    # and we need to handle negative variance (Full Truncation or Reflection).
    # Using Full Truncation scheme (Lord et al).
    
    for t in range(num_steps):
        # Euler discretization for Variance
        # dv = kappa * (theta - v) * dt + xi * sqrt(v) * dW2
        
        v_t = v[:, t]
        v_t_plus = np.maximum(v_t, 0) # Truncation for process calc
        
        dv = kappa * (theta - v_t_plus) * dt + xi * np.sqrt(v_t_plus * dt) * Z2[:, t]
        v_next = v_t + dv
        
        # Apply truncation for next step storage? 
        # Usually we store the actual process value but use truncated for diffusion.
        # But for price update we need valid vol.
        v[:, t+1] = v_next
        
        # Euler for Price
        # dS = r * S * dt + sqrt(v) * S * dW1
        
        # Use log price for stability?
        # d(ln S) = (r - 0.5 * v) * dt + sqrt(v) * dW1
        
        S_t = S[:, t]
        # v_t for price process: use same truncation
        v_t_eff = np.maximum(v_t, 0)
        
        d_ln_S = (r - 0.5 * v_t_eff) * dt + np.sqrt(v_t_eff * dt) * Z1[:, t]
        S[:, t+1] = S_t * np.exp(d_ln_S)
        
    # Payoff
    ST = S[:, -1]
    if option_type == 'C':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, S, v
