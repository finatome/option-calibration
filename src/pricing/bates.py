import numpy as np
from scipy.integrate import quad
from src.pricing.heston import heston_char_func

def merton_jump_char_func(u, T, lamb, mu, delta):
    """
    Adjusted Characteristic function for Merton '76 model: Only jump component
    Matches notebook L4 implementation.
    """
    # Characteristic exponent for the jump part
    # omega = E[e^J] - 1  (compensator for martingale)
    # in notebook: omega = -lamb * (np.exp(mu + 0.5 * delta**2) - 1)
    # This seems to be the drift correction term. 
    # The actual CF for jumps is exp(T * lambda * (phi_jump(u) - 1))
    # where phi_jump(u) is CF of jump size N(mu, delta).
    
    # Notebook implementation:
    omega = -lamb * (np.exp(mu + 0.5 * delta**2) - 1)
    char_func_value = np.exp(
        (1j * u * omega + lamb * (np.exp(1j * u * mu - u**2 * delta**2 * 0.5) - 1))
        * T
    )
    return char_func_value

def bates_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Bates (1996) characteristic function
    """
    # Heston component
    H93 = heston_char_func(u, T, r, kappa_v, theta_v, sigma_v, rho, v0)
    
    # Merton Jump component
    M76J = merton_jump_char_func(u, T, lamb, mu, delta)
    
    return H93 * M76J

def bates_int_func(u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta):
    """
    Lewis (2001) integral value for Bates (1996) characteristic function
    """
    char_func_value = bates_char_func(
        u - 1j * 0.5, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
    )
    int_func_value = (
        1 / (u**2 + 0.25) * (np.exp(1j * u * np.log(S0 / K)) * char_func_value).real
    )
    return int_func_value

def bates_price(S0, K, T, r, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, option_type='C'):
    """
    Valuation of European call option in Bates (1996) Model via Lewis (2001)
    
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
    - lamb: Jump intensity (lambda)
    - mu_j: Expected jump size (mu)
    - sigma_j: Jump size standard deviation (delta)
    - option_type: 'C' or 'P'
    """
    # Map parameters to internal names
    sigma_v = xi
    kappa_v = kappa
    theta_v = theta
    mu = mu_j
    delta = sigma_j
    
    # Integration limit from notebook/heston.py
    limit = 250 
    
    int_value = quad(
        lambda u: bates_int_func(
            u, S0, K, T, r, kappa_v, theta_v, sigma_v, rho, v0, lamb, mu, delta
        ),
        0,
        np.inf,
        limit=limit,
    )[0]
    
    call_value = max(0, S0 - np.exp(-r * T) * np.sqrt(S0 * K) / np.pi * int_value)
    
    if option_type.upper().startswith('P'):
        # Put-Call Parity
        # C - P = S0 - K * exp(-rT)
        # P = C - S0 + K * exp(-rT)
        put_value = call_value - S0 + K * np.exp(-r * T)
        return put_value
        
    return call_value
