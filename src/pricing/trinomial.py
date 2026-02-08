import numpy as np

def trinomial_tree_pricing(S_ini, K, T, r, sigma, N, option_type='C', style='European'):
    """
    Price an option using the Trinomial Tree model.
    Using the standard Kamrad-Ritchken or similar recombination logic.
    u = exp(sigma * sqrt(2*dt))
    d = 1/u
    m = 1
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(2 * dt))
    d = 1 / u
    m = 1.0
    
    # Probabilities
    # pu = ((exp(r*dt/2) - exp(-sigma*sqrt(dt/2))) / (exp(sigma*sqrt(dt/2)) - exp(-sigma*sqrt(dt/2))))^2
    # Simplified approach for small dt often used:
    # pu = ((exp(r*dt) - d) - (m-d)) / (u-d) ? No, that's binomial.
    
    # Using standard approximation:
    drift = r - 0.5 * sigma**2
    dx = sigma * np.sqrt(2 * dt)
    
    pu = ((np.exp(r * dt / 2) - np.exp(-dx / 2)) / (np.exp(dx / 2) - np.exp(-dx / 2))) ** 2
    pd = ((np.exp(dx / 2) - np.exp(r * dt / 2)) / (np.exp(dx / 2) - np.exp(-dx / 2))) ** 2
    pm = 1 - pu - pd
    
    # Correction: The above are for a specific tree type. Let's use the standard efficient set:
    # u = exp(lambda * sigma * sqrt(dt)), usually lambda approx sqrt(3) or similar for stability.
    # But let's stick to the simpler u, d, m = 1 structure if possible, or the one matching B-S moments.
    
    # Alternative accurate probabilities (Hull):
    u = np.exp(sigma * np.sqrt(3 * dt))
    d = 1 / u
    m = 1
    
    sqrt_dt = np.sqrt(dt/12) # This looks like specific implementation detail.
    
    # Let's use the robust parameters:
    dx = sigma * np.sqrt(3 * dt)
    u = np.exp(dx)
    d = np.exp(-dx)
    m = 1
    
    a = r * dt
    j_max = N # Nodes go from -N to +N
    
    pu = 0.5 * ( (sigma**2 * dt + (r - 0.5*sigma**2)**2 * dt**2) / dx**2 + (r - 0.5*sigma**2)*dt/dx )
    pd = 0.5 * ( (sigma**2 * dt + (r - 0.5*sigma**2)**2 * dt**2) / dx**2 - (r - 0.5*sigma**2)*dt/dx )
    pm = 1 - pu - pd

    # Initialize Tree (Size is 2N+1 because we expand up and down)
    # Mapping: j goes from -N to N. Array index [0] to [2N].
    # Center is index N.
    
    # Price Tree C[time_step, node_index]
    # To save space we can iterate backwards on a single vector, but for viz we might want the matrix.
    # Let's use matrix for consistency with binomial visualization.
    # Max width is 2*N + 1
    
    C = np.zeros((N + 1, 2 * N + 1))
    S = np.zeros((N + 1, 2 * N + 1))
    Delta = np.zeros((N, 2 * N + 1))  # Delta matrix (up to step N-1)
    
    N_steps = N # preserve N for referencing
    
    # Terminal Condition (t=N)
    for j in range(-N, N + 1):
        idx = j + N
        S[N, idx] = S_ini * np.exp(j * dx)
        if option_type == 'C':
            C[N, idx] = max(S[N, idx] - K, 0)
        else:
            C[N, idx] = max(K - S[N, idx], 0)
            
    # Backward Induction
    for i in range(N - 1, -1, -1):
        for j in range(-i, i + 1):
            idx = j + N
            
            # Underlying price at this node (re-calculate or store)
            S[i, idx] = S_ini * np.exp(j * dx)
            
            # Expected value
            # Neighbors at i+1 are j+1 (u), j (m), j-1 (d) ? 
            # Note: with u=e^dx, d=e^-dx, j increments by 1.
            # Next nodes are: 
            #   Up: j+1 -> idx+1 (relative to center? No, indices shift)
            #   Wait, if j is steps up from center.
            #   At step i, ranges from -i to i.
            #   Step i+1 ranges from -(i+1) to (i+1). 
            #   Node j at step i goes to j+1, j, j-1 at step i+1.
            
            # Array indices:
            # Current j corresponds to idx = j + N.
            # Next step (i+1), 'up' is j+1 => (j+1) + N = idx + 1
            # 'middle' is j => j + N = idx
            # 'down' is j-1 => (j-1) + N = idx - 1
            
            val_u = C[i+1, idx+1]
            val_m = C[i+1, idx]
            val_d = C[i+1, idx-1]
            
            continuation = np.exp(-r * dt) * (pu * val_u + pm * val_m + pd * val_d)
            
            if style == 'American':
                if option_type == 'C':
                    intrinsic = max(S[i, idx] - K, 0)
                else:
                    intrinsic = max(K - S[i, idx], 0)
                C[i, idx] = max(continuation, intrinsic)
            else:
                C[i, idx] = continuation
            
            # Delta Calculation (approximated using central difference of next nodes)
            # Delta = (V_u - V_d) / (S_u - S_d)
            # This is valid because at step i+1, nodes range from -(i+1) to +(i+1)
            # and our current j is in -i to i, so j+1 and j-1 are always valid at i+1
            if S[i+1, idx+1] != S[i+1, idx-1]:
                Delta[i, j + N_steps] = (C[i+1, idx+1] - C[i+1, idx-1]) / (S[i+1, idx+1] - S[i+1, idx-1])
            else:
                Delta[i, j + N_steps] = 0 # Should not happen in standard tree
                
    return C[0, N_steps], C, S, Delta
