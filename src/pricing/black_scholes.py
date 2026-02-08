import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='C'):
    """
    Calculate the Black-Scholes-Merton option price.

    Parameters:
    - S: Current stock price
    - K: Strike price
    - T: Time to maturity (years)
    - r: Risk-free rate (annual)
    - sigma: Volatility (annual)
    - option_type: 'C' or 'P'

    Returns:
    - price: Theoretical option price
    - greeks: Dictionary containing Delta, Gamma, Theta, Vega, Rho
    """
    # Handle scalar case specifically for efficiency and to avoid ambiguity errors
    if np.ndim(T) == 0:
         if T <= 0:
            if option_type == 'C':
                return max(S - K, 0), {}
            else:
                return max(K - S, 0), {}

    # For arrays, we proceed with vectorized calculation.
    # Note: If T <= 0 in array elements, this will generate warnings/NaNs unless handled.
    # Assuming caller provides valid T > 0 for now as per traceback context (T_rem + 1e-9).
    
    # Ensure inputs are numpy arrays for consistent broadcasting
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
        theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                 + r * K * np.exp(-r * T) * norm.cdf(-d2))

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    greeks = {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }

    return price, greeks

def black_scholes_mc(S0, K, T, r, sigma, option_type='C', num_paths=100, num_steps=100):
    """
    Monte Carlo Simulation for Black-Scholes Model.
    Returns the paths and the estimated price.
    """
    dt = T / num_steps
    # Generate random Brownian Motion
    Z = np.random.normal(0, 1, (num_paths, num_steps))
    
    # Initialize price paths
    S = np.zeros((num_paths, num_steps + 1))
    S[:, 0] = S0
    
    # Vectorized simulation
    # S_t = S_{t-1} * exp((r - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    
    # Cumulative sum of exponents for efficiency (S_t = S_0 * exp(cumsum(...)))
    # But calculating step-by-step for clarity matching standard Euler discretization
    # Actually, S_t = S_0 * exp( (r - 0.5*sigma^2)*t + sigma*W_t )
    # Let's use the cumulative sum method for speed and vectorization
    
    log_returns = drift + diffusion
    cumulative_returns = np.cumsum(log_returns, axis=1)
    S[:, 1:] = S0 * np.exp(cumulative_returns)
    
    # Calculate Payoff at probability T (last step)
    ST = S[:, -1]
    if option_type == 'C':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, S
