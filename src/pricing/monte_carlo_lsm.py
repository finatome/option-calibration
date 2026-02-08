import numpy as np

def lsmc_american_price(paths, K, r, T, option_type='P', poly_degree=2):
    """
    Price American Option using Longstaff-Schwartz (Least Squares Monte Carlo) method.
    
    Parameters:
    - paths: (num_paths, num_steps+1) array of asset prices.
    - K: Strike price.
    - r: Risk-free rate (annual).
    - T: Time to maturity (years).
    - option_type: 'C' or 'P'.
    - poly_degree: Degree of polynomial for regression (default 2: Laguerre-ish basis).
    
    Returns:
    - price: Estimated American option price.
    """
    S = paths
    num_paths, num_steps_plus_1 = S.shape
    num_steps = num_steps_plus_1 - 1
    dt = T / num_steps
    
    # Cash flow matrix (realized payoffs at exercise)
    cash_flows = np.zeros(num_paths)
    
    # Determine immediate payoff function
    if option_type == 'P':
        payoff_fn = lambda s: np.maximum(K - s, 0)
    else:
        payoff_fn = lambda s: np.maximum(s - K, 0)
        
    # Initialize cash flows at maturity
    cash_flows = payoff_fn(S[:, -1])
    
    # Discount factor per step
    df = np.exp(-r * dt)
    
    # Backward induction
    for t in range(num_steps - 1, 0, -1):
        # Discount cash flows from next step
        cash_flows = cash_flows * df
        
        # Current asset prices
        S_t = S[:, t]
        
        # In-the-money paths (only these are relevant for exercise decision)
        itm_mask = payoff_fn(S_t) > 0
        
        # If no paths are ITM, continue (no exercise possible)
        if np.count_nonzero(itm_mask) == 0:
            continue
            
        S_itm = S_t[itm_mask]
        cash_flows_itm = cash_flows[itm_mask]
        
        # Regression: Continuation Value ~ Polynomial(S_t)
        # E[Y | X] where Y = discounted future cash flow, X = current S
        # Using simple polynomial basis (1, S, S^2)
        regression = np.polyfit(S_itm, cash_flows_itm, poly_degree)
        continuation_value = np.polyval(regression, S_itm)
        
        # Immediate exercise value
        exercise_value = payoff_fn(S_itm)
        
        # Identify paths where exercise is optimal
        exercise_mask = exercise_value > continuation_value
        
        # Update cash flows for those who exercised
        # We need to map back to the original full array
        # Create a full update mask
        full_exercise_mask = np.zeros(num_paths, dtype=bool)
        full_exercise_mask[itm_mask] = exercise_mask
        
        # Update cash flows: If exercised, value is exercise_value.
        # If not exercised, value remains discounted future cash flow.
        cash_flows[full_exercise_mask] = payoff_fn(S[full_exercise_mask, t])
        
    # Discount back to time 0
    price = np.mean(cash_flows * df)
    
    return price
