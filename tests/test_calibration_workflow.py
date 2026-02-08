import unittest
import numpy as np
import pandas as pd
from src.data.synthetic import generate_option_surface
from src.calibration.optimize import calibrate_model

class TestCalibrationWorkflow(unittest.TestCase):
    
    def test_heston_calibration_recovery(self):
        """
        Test if Heston calibration can recover known parameters from synthetic data.
        """
        # 1. Define True Parameters
        true_params = {
            'v0': 0.04,
            'kappa': 2.0,
            'theta': 0.04,
            'xi': 0.3,
            'rho': -0.5
        }
        
        # 2. Generate Synthetic Data
        # Use a dense grid to ensure good recovery
        df = generate_option_surface(
            S0=100.0, r=0.05,
            min_strike=0.8, max_strike=1.2, num_strikes=5,
            min_expiry=0.5, max_expiry=1.5, num_expiries=3,
            model_name='Heston',
            model_params=true_params,
            noise_level=0.0 # No noise for exact recovery test
        )
        
        # 3. Calibrate
        # Initial guess slightly off
        initial_guess = [0.04, 1.0, 0.04, 0.2, -0.2] 
        # v0, kappa, theta, xi, rho
        
        result = calibrate_model('Heston', df, initial_guess=initial_guess)
        
        # 4. Verify
        print("\nCalibration Result:")
        print(f"Success: {result['success']}")
        print(f"Message: {result['message']}")
        print(f"Error: {result['error']}")
        print(f"Params: {result['params']}")
        
        # Check if recovered params are close to true params
        # v0
        self.assertAlmostEqual(result['params'][0], true_params['v0'], delta=0.01)
        # kappa (hard to recover exactly, usually loose tolerance)
        self.assertAlmostEqual(result['params'][1], true_params['kappa'], delta=1.5) 
        # theta (long term vol)
        self.assertAlmostEqual(result['params'][2], true_params['theta'], delta=0.01)
        # xi (vol of vol)
        self.assertAlmostEqual(result['params'][3], true_params['xi'], delta=0.1)
        # rho (correlation)
        self.assertAlmostEqual(result['params'][4], true_params['rho'], delta=0.2)

if __name__ == '__main__':
    unittest.main()
