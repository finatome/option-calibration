# Put-Call Parity

Put-Call Parity is a fundamental principle in financial engineering that defines the relationship between the price of a European Call option ($C$) and a European Put option ($P$) with the same strike ($K$) and expiration ($T$).

## The Equation
For non-dividend paying stocks:

$$ C + K e^{-rT} = P + S_0 $$

Where:
-   $C$: Call Option Price
-   $K e^{-rT}$: Present Value of the Strike Price (Cash)
-   $P$: Put Option Price
-   $S_0$: Current Spot Price of the Asset

## Intuition
This relationship relies on a no-arbitrage argument.
-   **Portfolio A**: Buy a Call + Cash equal to PV(K).
-   **Portfolio B**: Buy a Put + The Stock.

At expiration $T$:
-   If $S_T > K$: Call is exercised ($S_T - K$). Cash becomes $K$. Total = $S_T$.
-   If $S_T < K$: Put is exercised ($K - S_T$). Stock is $S_T$. Total = $K$.
    
Both portfolios have the exact same payoff in all states. Therefore, they must have the same price today.

## Verification
The dashboard automatically verifies this condition.
1.  When you calculate a **European** option price, the system implicitly calculates the corresponding option (Put for Call, Call for Put).
2.  It checks the difference: $(C + Ke^{-rT}) - (P + S_0)$.
3.  If the difference is close to zero, Parity Holds.

> **Note**: This relationship does **not** strictly hold for American options due to the early exercise premium.
