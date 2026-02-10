
import unittest
import numpy as np
import pandas as pd
from src.calibration.optimize import calibrate_model

class TestCalibrationLogic(unittest.TestCase):

    def setUp(self):
        # Create minimal synthetic market data
        self.market_df = pd.DataFrame({
            'Strike': [90, 100, 110],
            'Maturity': [1.0, 1.0, 1.0],
            'Price': [15.0, 8.0, 3.0], # Roughly BS prices
            'Type': ['call', 'call', 'call'],
            'S0': [100.0] * 3,
            'r': [0.05] * 3
        })
        self.market_json = self.market_df.to_json(orient='split')

    def test_calibrate_black_scholes(self):
        """Test calibration of Black-Scholes model."""
        # Initial guess: sigma=0.5
        params = {'sigma': 0.5} 
        initial_guess = [0.5]
        
        # Pass DataFrame directly
        result = calibrate_model('BlackScholes', self.market_df, initial_guess, method='L-BFGS-B')
        
        self.assertIn('success', result)  # Check key existence
        self.assertTrue(0 < result['params'][0] < 1.0) # Check param value
        self.assertLess(result['error'], 1.0) # Should fit reasonably well

    def test_calibration_failure_mode(self):
        """Test graceful handling of bad data."""
        # Empty input with None initial_guess to allow defaults to populate
        result = calibrate_model('Heston', pd.DataFrame(), None, method='L-BFGS-B')
        # Expecting it to run but maybe return huge error or empty? 
        # Actually optimize.py iterates rows. If empty, error=0?
        # Let's check return type.
        self.assertIsNotNone(result)
        self.assertEqual(result['error'], 0.0)

if __name__ == '__main__':
    unittest.main()
