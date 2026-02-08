import numpy as np
import pandas as pd
from src.pricing.heston import heston_price
from src.pricing.merton import merton_jump_diffusion_price
from src.pricing.black_scholes import black_scholes_price

def generate_option_surface(
    S0=100.0,
    r=0.05,
    min_strike=0.8,
    max_strike=1.2,
    num_strikes=10,
    min_expiry=0.1,
    max_expiry=2.0,
    num_expiries=5,
    model_name='Heston',
    model_params=None,
    noise_level=0.0
):
    """
    Generates a grid of option prices based on a given model and parameters.
    
    Parameters:
    - S0: Spot Price
    - r: Risk-free rate
    - min_strike, max_strike: Range of moneyness (Strike/S0)
    - num_strikes: Number of strikes in the grid
    - min_expiry, max_expiry: Range of maturities (Years)
    - num_expiries: Number of maturities in the grid
    - model_name: 'Heston', 'Merton', or 'BlackScholes'
    - model_params: Dictionary of model parameters
    - noise_level: Percentage of Gaussian noise to add to prices (e.g. 0.01 for 1%)
    
    Returns:
    - DataFrame with columns ['Strike', 'Maturity', 'Type', 'Price', 'True_Price', 'IV']
    """
    
    # Generate Grid
    strikes = np.linspace(min_strike * S0, max_strike * S0, num_strikes)
    expiries = np.linspace(min_expiry, max_expiry, num_expiries)
    
    data = []
    
    # Defaults
    if model_params is None:
        model_params = {}

    for T in expiries:
        for K in strikes:
            # We focus on Calls for calibration usually, or OTM Puts / OTM Calls. 
            # For simplicity, let's generate Calls.
            option_type = 'C'
            
            price = 0.0
            
            if model_name == 'Heston':
                # Heston Params
                v0 = model_params.get('v0', 0.04)
                kappa = model_params.get('kappa', 2.0)
                theta = model_params.get('theta', 0.04)
                xi = model_params.get('xi', 0.3)
                rho = model_params.get('rho', -0.5)
                
                price = heston_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type)
                
            elif model_name == 'Merton':
                # Merton Params
                sigma = model_params.get('sigma', 0.2)
                lambda_j = model_params.get('lambda_j', 1.0)
                mu_j = model_params.get('mu_j', -0.1)
                sigma_j = model_params.get('sigma_j', 0.1)
                
                price = merton_jump_diffusion_price(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, option_type)
                
            elif model_name == 'BlackScholes':
                sigma = model_params.get('sigma', 0.2)
                price, _ = black_scholes_price(S0, K, T, r, sigma, option_type)
            
            # Add Noise
            noise = 0.0
            if noise_level > 0:
                noise = price * noise_level * np.random.normal()
            
            market_price = max(0.0, price + noise) # Prices can't be negative
            
            data.append({
                'Strike': K,
                'Maturity': T,
                'Type': option_type,
                'Price': market_price,
                'True_Price': price,
                'S0': S0,
                'r': r
            })
            
    return pd.DataFrame(data)
