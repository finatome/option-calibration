import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pricing.bates import bates_price
from src.pricing.heston import heston_price
from src.pricing.black_scholes import black_scholes_price

def test_bates_equals_heston_no_jumps():
    """
    If lambda is 0, Bates price should equal Heston price.
    """
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    v0 = 0.04
    kappa = 1.5
    theta = 0.04
    xi = 0.3
    rho = -0.5
    
    # Bates with no jumps
    lamb = 0.0
    mu_j = -0.1
    sigma_j = 0.1
    
    bates_val = bates_price(S0, K, T, r, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, option_type='C')
    heston_val = heston_price(S0, K, T, r, v0, kappa, theta, xi, rho, option_type='C')
    
    print(f"Bates (lambda=0): {bates_val}")
    print(f"Heston: {heston_val}")
    
    assert np.isclose(bates_val, heston_val, atol=1e-4)

def test_bates_random_case():
    """
    Test a random case to ensure it doesn't crash.
    """
    S0 = 100
    K = 110
    T = 0.5
    r = 0.02
    v0 = 0.04
    kappa = 2.0
    theta = 0.04
    xi = 0.4
    rho = -0.7
    lamb = 1.0
    mu_j = 0.0
    sigma_j = 0.2
    
    price = bates_price(S0, K, T, r, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j)
    assert price > 0

if __name__ == '__main__':
    test_bates_equals_heston_no_jumps()
    test_bates_random_case()
    print("All tests passed!")
