import numpy as np
from src.pricing.black_scholes import black_scholes_price

def merton_jump_diffusion_price(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, option_type='C', max_steps=50):
    """
    Calculate Merton Jump Diffusion Option Price using series of Black-Scholes prices.
    
    Parameters:
    - lambda_j: Jump intensity (expected number of jumps per year)
    - mu_j: Mean of log jump size
    - sigma_j: Standard deviation of log jump size
    """
    # Expected percentage change from a jump
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    
    # Adjusted drift parameters to maintain risk-neutrality
    # The drift of the diffusions part is r - lambda * k - 0.5 * sigma^2
    # But for the formula, we adjust r and sigma for each term n (number of jumps)
    
    # lambda' = lambda * (1+k)
    lambda_prime = lambda_j * (1 + k)
    
    price = 0
    
    # Calculate terms
    # We truncate the infinite series when terms become negligible or max_steps reached
    
    for n in range(max_steps):
        # Factor for n jumps
        # P(N(T) = n) under measure Q' ?
        # Actually simplest form:
        # P = sum_{n=0}^{inf} exp(-lambda' * T) * (lambda' * T)^n / n! * BS(...)
        
        # Calculate r_n and sigma_n
        # r_n = r - lambda * k + n * log(1+k) / T
        
        if np.ndim(T) == 0:
            if T > 0:
                r_n = r - lambda_j * k + n * np.log(1 + k) / T
                sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / T)
            else:
                 r_n = r
                 sigma_n = sigma
        else:
             # Vectorized case: assumes T is well-behaved (e.g. T > 1e-9)
             # or handle T=0 using np.where if needed.
             # Given caller context (T_rem + 1e-9), we can compute directly.
             # To be robust against strict 0 in array:
             values_safe = np.maximum(T, 1e-9) 
             r_n = r - lambda_j * k + n * np.log(1 + k) / values_safe
             sigma_n = np.sqrt(sigma**2 + n * sigma_j**2 / values_safe)
             
        # Probability weight
        # exp(-lambda' * T) * (lambda' * T)^n / n!
        
        import math
        weight = np.exp(-lambda_prime * T) * (lambda_prime * T)**n / math.factorial(n)
        
        bs_p, _ = black_scholes_price(S0, K, T, r_n, sigma_n, option_type.upper())
        
        price += weight * bs_p
        
        if np.all(weight < 1e-10) and n > 5:
            break
            
    return price

def merton_jump_diffusion_mc(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, option_type='C', num_paths=100, num_steps=100):
    """
    Monte Carlo Simulation for Merton Jump Diffusion Model.
    """
    dt = T / num_steps
    
    # Generate geometric Brownian motion parts
    Z = np.random.normal(0, 1, (num_paths, num_steps))
    
    # Jump component
    # Number of jumps in each step: Poisson(lambda * dt)
    # Simulation of jumps can be done by simulating N ~ Poisson(lambda * T) total jumps and placing them,
    # or step-by-step Bernoulli approximation if lambda*dt is small, or actual Poisson.
    
    # Let's use Poisson for each step
    J = np.random.poisson(lambda_j * dt, (num_paths, num_steps))
    
    # Jump sizes
    # We need a sum of J normals.
    # sum_Y = sum_{i=1}^J Y_i, Y_i ~ N(mu_j, sigma_j^2)
    # This sum is distributed as N(J * mu_j, J * sigma_j^2)
    
    # Careful: if J=0, sum is 0.
    
    # We can simulate the total jump size for each step directly
    # jump_size_log = N(J*mu_j, J*sigma_j^2)
    
    # Create mask for steps with jumps
    has_jumps = J > 0
    
    jump_log_return = np.zeros((num_paths, num_steps))
    
    # For paths/steps with jumps, generate random normal
    # We can just generate for all and multiply by correctness, 
    # but variance depends on J.
    # Vectorized approach:
    # J is matrix of integers.
    
    # We calculate mean and std for the jump part at each step
    jump_means = J * mu_j
    jump_stds = np.sqrt(J) * sigma_j
    
    # Generate standard normal for jump magnitude
    W_jump = np.random.normal(0, 1, (num_paths, num_steps))
    
    jump_log_return = np.where(has_jumps, jump_means + jump_stds * W_jump, 0)
    
    # Drift correction
    # k = E[e^Y - 1]
    k = np.exp(mu_j + 0.5 * sigma_j**2) - 1
    drift_correction = -lambda_j * k * dt
    
    # Regular GBM part
    gbm_log_return = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    
    # Total Log Return
    total_log_return = gbm_log_return + drift_correction + jump_log_return
    
    # Cumulative Return
    cumulative_returns = np.cumsum(total_log_return, axis=1)
    
    S = np.zeros((num_paths, num_steps + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(cumulative_returns)
    
    # Payoff
    ST = S[:, -1]
    if option_type == 'C':
        payoffs = np.maximum(ST - K, 0)
    else:
        payoffs = np.maximum(K - ST, 0)
        
    price = np.exp(-r * T) * np.mean(payoffs)
    
    return price, S
