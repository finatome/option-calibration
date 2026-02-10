
import unittest
import numpy as np
from src.pricing.heston import heston_price
from src.pricing.merton import merton_jump_diffusion_price
from src.pricing.black_scholes import black_scholes_price
from src.pricing.bates import bates_price

class TestPricingEngines(unittest.TestCase):

    def setUp(self):
        # Standard test parameters
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        
        # Heston Parameters
        self.v0 = 0.04
        self.kappa = 2.0
        self.theta = 0.04
        self.xi = 0.1
        self.rho = -0.5
        
        # Merton Parameters
        self.sigma = 0.2
        self.lam = 1.0
        self.mu_j = -0.1
        self.sigma_j = 0.1
        
        # Bates Parameters (Combine Heston + Merton)
        self.bates_params = (self.v0, self.kappa, self.theta, self.xi, self.rho, self.lam, self.mu_j, self.sigma_j)

    def test_heston_call_put_parity(self):
        """Verify Heston model satisfies Put-Call Parity."""
        # Signature: S0, K, T, r, v0, kappa, theta, xi, rho, option_type
        call_price = heston_price(self.S0, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, 'call')
        put_price = heston_price(self.S0, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, 'put')
        
        # C - P = S - K*exp(-rT)
        lhs = call_price - put_price
        rhs = self.S0 - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=4, msg="Heston Put-Call Parity failed")

    def test_merton_call_put_parity(self):
        """Verify Merton model satisfies Put-Call Parity."""
        call_price = merton_jump_diffusion_price(self.S0, self.K, self.T, self.r, self.sigma, self.lam, self.mu_j, self.sigma_j, 'call')
        put_price = merton_jump_diffusion_price(self.S0, self.K, self.T, self.r, self.sigma, self.lam, self.mu_j, self.sigma_j, 'put')
        
        lhs = call_price - put_price
        rhs = self.S0 - self.K * np.exp(-self.r * self.T)
        self.assertAlmostEqual(lhs, rhs, places=4, msg="Merton Put-Call Parity failed")

    def test_bates_limits(self):
        """Verify Bates converges to Heston when jumps are zero."""
        # Bates with lambda=0 should imply simple Heston
        # Bates sig: S0, K, T, r, v0, kappa, theta, xi, rho, lamb, mu, delta
        bates_call = bates_price(self.S0, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, 0, 0, 0, 'call')
        # Heston sig: S0, K, T, r, v0, kappa, theta, xi, rho
        heston_call = heston_price(self.S0, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, 'call')
        
        self.assertAlmostEqual(bates_call, heston_call, places=4, msg="Bates (lambda=0) != Heston")

    def test_pricing_consistency(self):
        """Ensure prices are non-negative and finite."""
        models = [
            (heston_price, (self.S0, self.K, self.T, self.r, self.v0, self.kappa, self.theta, self.xi, self.rho, 'call')),
            (merton_jump_diffusion_price, (self.S0, self.K, self.T, self.r, self.sigma, self.lam, self.mu_j, self.sigma_j, 'call')),
            (bates_price, (self.S0, self.K, self.T, self.r, *self.bates_params, 'call'))
        ]
        
        for price_func, args in models:
            price = price_func(*args)
            self.assertTrue(price > 0, f"{price_func.__name__} returned non-positive price")
            self.assertFalse(np.isnan(price), f"{price_func.__name__} returned NaN")

if __name__ == '__main__':
    unittest.main()
