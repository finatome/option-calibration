import numpy as np
from src.pricing.black_scholes import black_scholes_price

def calculate_numerical_greeks(pricing_func, S, K, T, r, sigma, option_type='C', style='European', **kwargs):
    """
    Calculate Greeks using finite difference method on any pricing function.
    """
    
    base_price = pricing_func(S, K, T, r, sigma, option_type=option_type, style=style, **kwargs)[0]
    
    # Delta & Gamma (Shock S)
    dS = S * 0.01
    price_up = pricing_func(S + dS, K, T, r, sigma, option_type=option_type, style=style, **kwargs)[0]
    price_down = pricing_func(S - dS, K, T, r, sigma, option_type=option_type, style=style, **kwargs)[0]
    
    delta = (price_up - price_down) / (2 * dS)
    gamma = (price_up - 2 * base_price + price_down) / (dS ** 2)
    
    # Vega (Shock Sigma)
    dSigma = 0.01
    price_vol_up = pricing_func(S, K, T, r, sigma + dSigma, option_type=option_type, style=style, **kwargs)[0]
    vega = (price_vol_up - base_price) / dSigma # Usually quoted per 1% change, or just pure deriv
    # Convention: Vega is change per 1% change usually? No, strictly deriv wrt sigma.
    # Often scaled by 0.01 in reporting. We'll return raw derivative.
    
    # Theta (Shock T)
    dt = 1/365.0
    if T > dt:
        price_t_down = pricing_func(S, K, T - dt, r, sigma, option_type=option_type, style=style, **kwargs)[0]
        theta = (price_t_down - base_price) / dt # Price change per year decay (usually negative)
    else:
        theta = 0
        
    # Rho (Shock r)
    dr = 0.01
    price_r_up = pricing_func(S, K, T, r + dr, sigma, option_type=option_type, style=style, **kwargs)[0]
    rho = (price_r_up - base_price) / dr

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho
    }
