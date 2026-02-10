import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.pricing.heston import heston_price, heston_char_func
from src.pricing.merton import merton_jump_diffusion_price
from src.pricing.black_scholes import black_scholes_price
from src.pricing.bates import bates_price, bates_char_func
from src.pricing.fft import carr_madan_price

def mean_squared_error(params, market_data, model_name, pricing_method='Lewis'):
    """
    Objective function for calibration.
    """
    error = 0.0
    N = len(market_data)
    if N == 0:
        return 0.0
    
    # Pre-computation for FFT if using Carr-Madan across full surface?
    if pricing_method == 'Carr-Madan' and model_name in ['Heston', 'Bates']:
        # Group by Maturity to optimize FFT calls
        unique_expiries = market_data['Maturity'].unique()
        total_sq_error = 0.0
        
        for T in unique_expiries:
            subset = market_data[market_data['Maturity'] == T]
            if subset.empty: continue
            
            S0 = subset.iloc[0]['S0']
            r = subset.iloc[0]['r']
            strikes = subset['Strike'].values
            market_prices = subset['Price'].values
            types = subset['Type'].values
            
            # Construct CF lambda based on params
            if model_name == 'Heston':
                v0, kappa, theta, xi, rho = params
                cf = lambda u: heston_char_func(u, T, r, kappa, theta, xi, rho, v0)
            elif model_name == 'Bates':
                v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j = params
                cf = lambda u: bates_char_func(u, T, r, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)
            
            # FFT Pricing (returns array matching strikes input size?)
            # The fft.py implementation interpolates to requested K.
            # It returns call prices. 
            model_prices_call = carr_madan_price(S0, strikes, T, r, cf)
            
            # Handle Puts using Parity if needed
            # Assuming FFT returns Call prices.
            
            # model_prices_call is np array
            
            for i in range(len(subset)):
                val = model_prices_call[i]
                mkt_price = market_prices[i]
                opt_type = types[i]
                
                mod_price = val
                if opt_type == 'P':
                    # C - P = S - K*e^-rT  =>  P = C - S + K*e^-rT
                    mod_price = val - S0 + strikes[i] * np.exp(-r * T)
                    
                total_sq_error += (mkt_price - mod_price)**2
                
        return total_sq_error / N

    
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

        elif model_name == 'Bates':
            # Params: v0, kappa, theta, xi, rho, lambda, mu_j, sigma_j
            v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j = params
            model_price = bates_price(S0, K, T, r, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, option_type)
            
        error += (market_price - model_price)**2
        
    return error / N

def calibrate_model(model_name, market_data, initial_guess=None, bounds=None, method='L-BFGS-B', pricing_method='Lewis'):
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
        elif model_name == 'Bates':
            # v0, kappa, theta, xi, rho, lambda, mu_j, sigma_j
            # Combining Heston and Merton defaults roughly
            initial_guess = [0.04, 1.5, 0.04, 0.3, -0.5, 0.1, -0.1, 0.1]

    if bounds is None:
        if model_name == 'Heston':
            bounds = [(0.001, 1.0), (0.01, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99)]
        elif model_name == 'Merton':
            bounds = [(0.01, 1.0), (0.0, 5.0), (-1.0, 1.0), (0.0, 1.0)]
        elif model_name == 'BlackScholes':
            bounds = [(0.01, 2.0)]
        elif model_name == 'Bates':
            bounds = [
                (0.001, 1.0), (0.01, 10.0), (0.001, 1.0), (0.01, 2.0), (-0.99, 0.99), # Heston
                (0.0, 5.0), (-1.0, 1.0), (0.0, 1.0) # Merton (lambda, mu, sigma_j)
            ]
            
    # Handle methods that don't support bounds if necessary, or let scipy handle error/warning
    # L-BFGS-B, TNC, SLSQP all support bounds. Nelder-Mead does not (ignores).
    
    result = minimize(
        mean_squared_error, 
        initial_guess, 
        args=(market_data, model_name, pricing_method), 
        method=method, 
        bounds=bounds if method in ['L-BFGS-B', 'TNC', 'SLSQP', 'Powell'] else None
    )
    
    return {
        'success': result.success,
        'message': result.message,
        'params': result.x.tolist(),
        'error': result.fun
    }
