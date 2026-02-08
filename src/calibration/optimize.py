import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.pricing.heston import heston_price
from src.pricing.merton import merton_jump_diffusion_price
from src.pricing.black_scholes import black_scholes_price

def mean_squared_error(params, market_data, model_name):
    """
    Objective function for calibration.
    """
    error = 0.0
    N = len(market_data)
    
    for index, row in market_data.iterrows():
        S0 = row['S0']
        K = row['Strike']
        T = row['Maturity']
        r = row['r']
        market_price = row['Price']
        option_type = row['Type']
        
        model_price = 0.0
        
        if model_name == 'Heston':
            # Params: v0, kappa, theta, xi, rho
            v0, kappa, theta, xi, rho = params
            model_price = heston_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type)
            
        elif model_name == 'Merton':
            # Params: sigma, lambda_j, mu_j, sigma_j
            sigma, lambda_j, mu_j, sigma_j = params
            model_price = merton_jump_diffusion_price(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, option_type)
            
        elif model_name == 'BlackScholes':
            # Params: sigma
            sigma = params[0]
            model_price, _ = black_scholes_price(S0, K, T, r, sigma, option_type)
            
        error += (market_price - model_price)**2
        
    return error / N

def calibrate_model(model_name, market_data, initial_guess=None, bounds=None, method='L-BFGS-B'):
    """
    Calibrates model parameters to market data.
    """
    
    if initial_guess is None:
        if model_name == 'Heston':
            # v0, kappa, theta, xi, rho
            initial_guess = [0.04, 1.5, 0.04, 0.3, -0.5]
        elif model_name == 'Merton':
            # sigma, lambda, mu_j, sigma_j
            initial_guess = [0.2, 1.0, -0.1, 0.1]
        elif model_name == 'BlackScholes':
            # sigma
            initial_guess = [0.2]

    if bounds is None:
        if model_name == 'Heston':
            bounds = [(0.001, 1.0), (0.01, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99)]
        elif model_name == 'Merton':
            bounds = [(0.01, 1.0), (0.0, 5.0), (-1.0, 1.0), (0.0, 1.0)]
        elif model_name == 'BlackScholes':
            bounds = [(0.01, 2.0)]
            
    # Handle methods that don't support bounds if necessary, or let scipy handle error/warning
    # L-BFGS-B, TNC, SLSQP all support bounds. Nelder-Mead does not (ignores).
    
    result = minimize(
        mean_squared_error, 
        initial_guess, 
        args=(market_data, model_name), 
        method=method, 
        bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP', 'Powell'] else None
    )
    
    return {
        'success': result.success,
        'message': result.message,
        'params': result.x.tolist(),
        'error': result.fun
    }
