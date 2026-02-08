import numpy as np

def check_put_call_parity(call_price, put_price, S, K, T, r, q=0):
    """
    Check if Put-Call Parity holds: C + K*exp(-rT) = P + S*exp(-qT)
    Returns a dictionary with details.
    """
    lhs = call_price + K * np.exp(-r * T)
    rhs = put_price + S * np.exp(-q * T)
    
    diff = lhs - rhs
    holds = np.isclose(diff, 0, atol=1e-2) # Tolerance of 1 cent
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'lhs': lhs,
        'rhs': rhs,
        'difference': diff,
        'holds': bool(holds)
    }
