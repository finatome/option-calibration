import unittest
import numpy as np
from src.pricing.black_scholes import black_scholes_price, black_scholes_mc
from src.pricing.heston import heston_price, heston_mc
from src.pricing.merton import merton_jump_diffusion_price, merton_jump_diffusion_mc

class TestContinuousModels(unittest.TestCase):
    
    def test_black_scholes_mc_convergence(self):
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        analytical, _ = black_scholes_price(S0, K, T, r, sigma, 'C')
        
        # MC with decent number of paths
        mc_price, _ = black_scholes_mc(S0, K, T, r, sigma, 'C', num_paths=10000, num_steps=100)
        
        # Check within 1% error (MC variance ~ 1/sqrt(N))
        self.assertAlmostEqual(mc_price, analytical, delta=analytical * 0.02)
        
    def test_heston_price_benchmark(self):
        # Parameters where Heston -> Black Scholes
        # sigma_BS = sqrt(v0) if kappa -> infinity or xi -> 0 and v0 = theta
        
        S0, K, T, r = 100, 100, 1.0, 0.05
        v0 = 0.04
        kappa = 1.0
        theta = 0.04
        xi = 0.001 # approx 0
        rho = 0.0
        
        bs_price, _ = black_scholes_price(S0, K, T, r, np.sqrt(v0), 'C')
        heston_p = heston_price(S0, K, T, r, v0, kappa, theta, xi, rho, 'C')
        
        self.assertAlmostEqual(heston_p, bs_price, delta=0.5)
        
    def test_merton_price_benchmark(self):
        # Parameters where Merton -> Black Scholes (lambda = 0)
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        lambda_j = 0.0
        mu_j = 0.0
        sigma_j = 0.0
        
        bs_price, _ = black_scholes_price(S0, K, T, r, sigma, 'C')
        merton_p = merton_jump_diffusion_price(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, 'C')
        
        self.assertAlmostEqual(merton_p, bs_price, places=4)
        
    def test_merton_mc_convergence(self):
        S0, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.2
        lambda_j = 1.0
        mu_j = -0.1
        sigma_j = 0.1
        
        analytical = merton_jump_diffusion_price(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, 'C')
        mc_price, _ = merton_jump_diffusion_mc(S0, K, T, r, sigma, lambda_j, mu_j, sigma_j, 'C', num_paths=10000)
        
        self.assertAlmostEqual(mc_price, analytical, delta=analytical * 0.05)

if __name__ == '__main__':
    unittest.main()
