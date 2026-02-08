import numpy as np

def binomial_tree_pricing(S_ini, K, T, r, sigma, N, option_type='C', style='European'):
    """
    Price an option using the Binomial Tree model.

    Parameters:
    - S_ini: Initial stock price
    - K: Strike price
    - T: Time to maturity (years)
    - r: Risk-free rate (annual)
    - sigma: Volatility (annual)
    - N: Number of time steps
    - option_type: 'C' for Call, 'P' for Put
    - style: 'European' or 'American'

    Returns:
    - price: Option price at t=0
    - C: Option price matrix
    - S: Underlying price matrix
    - Delta: Delta matrix
    """
    dt = T / N  # Time step
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor (using CRR specification usually, or d = 1/u)
    # Check if we should use the definition from the notebook: d = exp(-sigma*sqrt(dt))
    # In the notebook: u = exp(sigma*sqrt(dt)), d = exp(-sigma*sqrt(dt)). This satisfies ud=1.
    
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Initialize matrices
    C = np.zeros([N + 1, N + 1])
    S = np.zeros([N + 1, N + 1])
    Delta = np.zeros([N, N])

    # 1. Build the Stock Price Tree (Forward)
    for i in range(0, N + 1):
        S[N, i] = S_ini * (u ** (i)) * (d ** (N - i))
        
        # Calculate terminal payoff
        if option_type == 'C':
            C[N, i] = max(S[N, i] - K, 0)
        else:
            C[N, i] = max(K - S[N, i], 0)

    # 2. Backward Induction for Option Price
    for j in range(N - 1, -1, -1):
        for i in range(0, j + 1):
            # European Value (Risk-Neutral Expectation)
            continuation_value = np.exp(-r * dt) * (p * C[j + 1, i + 1] + (1 - p) * C[j + 1, i])
            
            # Underlying price at this node
            S[j, i] = S_ini * (u ** (i)) * (d ** (j - i))
            
            # Determine Option Value
            if style == 'American':
                if option_type == 'C':
                    intrinsic_value = max(S[j, i] - K, 0)
                else:
                    intrinsic_value = max(K - S[j, i], 0)
                C[j, i] = max(continuation_value, intrinsic_value)
            else:
                C[j, i] = continuation_value
            
            # Calculate Delta (Sensitivity)
            # Delta = (C_up - C_down) / (S_up - S_down)
            if j < N: # Ensure we are not at the very end where delta is undefined or needs different calc
                 # S[j+1, i+1] is S_up (more u's), S[j+1, i] is S_down
                denom = (S[j + 1, i + 1] - S[j + 1, i])
                if denom == 0:
                    Delta[j, i] = 0
                else:
                    Delta[j, i] = (C[j + 1, i + 1] - C[j + 1, i]) / denom

    return C[0, 0], C, S, Delta
