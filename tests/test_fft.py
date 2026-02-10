import sys
import os
import unittest
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.pricing.fft import carr_madan_price
from src.pricing.heston import heston_char_func, heston_price
from src.pricing.bates import bates_char_func, bates_price

class TestCarrMadan(unittest.TestCase):
    
    def test_carr_madan_heston_consistency(self):
        """Test if Carr-Madan FFT matches Lewis integration for Heston"""
        S0 = 100
        K = [90, 100, 110]
        T = 1.0
        r = 0.05
        
        # Heston Params
        v0, kappa, theta, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.5
        
        # Define CF for FFT
        cf = lambda u: heston_char_func(u, T, r, kappa, theta, xi, rho, v0)
        
        # Calculate FFT prices
        fft_prices = carr_madan_price(S0, K, T, r, cf)
        
        # Calculate Lewis prices
        lewis_prices = []
        for k in K:
            lewis_prices.append(heston_price(S0, k, T, r, v0, kappa, theta, xi, rho, 'C'))
            
        print(f"\nHeston FFT: {fft_prices}")
        print(f"Heston Lewis: {lewis_prices}")
        
        # Allow some difference due to discretization/methods
        np.testing.assert_allclose(fft_prices, lewis_prices, rtol=1e-2, atol=1e-2)

    def test_carr_madan_bates_consistency(self):
        """Test if Carr-Madan FFT matches Lewis integration for Bates"""
        S0 = 100
        K = [90, 100, 110]
        T = 1.0
        r = 0.05
        
        # Bates Params
        v0, kappa, theta, xi, rho = 0.04, 1.5, 0.04, 0.3, -0.5
        lamb, mu_j, sigma_j = 0.1, -0.1, 0.1
        
        # Define CF for FFT
        cf = lambda u: bates_char_func(u, T, r, kappa, theta, xi, rho, v0, lamb, mu_j, sigma_j)
        
        # Calculate FFT prices
        fft_prices = carr_madan_price(S0, K, T, r, cf)
        
        # Calculate Lewis prices
        lewis_prices = []
        for k in K:
            lewis_prices.append(bates_price(S0, k, T, r, v0, kappa, theta, xi, rho, lamb, mu_j, sigma_j, 'C'))
            
        print(f"\nBates FFT: {fft_prices}")
        print(f"Bates Lewis: {lewis_prices}")
        
        np.testing.assert_allclose(fft_prices, lewis_prices, rtol=1e-2, atol=1e-2)

if __name__ == '__main__':
    unittest.main()
