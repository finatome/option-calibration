# Delta Hedging

Delta Hedging is a strategy used to reduce the risk associated with price movements in an underlying asset. By offsetting long and short positions, a portfolio can become "Delta Neutral," meaning its value remains unchanged for small movements in the underlying asset's price.

## Concept

The **Delta ($\Delta$)** of an option represents the number of shares of the underlying stock that must be held to replicate the option's payoff for small price changes.
-   **Long Call**: Positive Delta (Need to own shares).
-   **Short Call**: Negative Delta (Need to short shares).

If a bank sells a Call Option (Short Call), they are exposed to the risk of the stock rising. To hedge this:
1.  Calculate $\Delta$.
2.  Buy $\Delta$ shares of the stock.
3.  Portfolio = $-1 \cdot C + \Delta \cdot S$.

For small movements $dS$, the change in portfolio value is:
$$ d\Pi = -1 \cdot dC + \Delta \cdot dS $$
Since $dC \approx \Delta \cdot dS$, then:
$$ d\Pi \approx -\Delta \cdot dS + \Delta \cdot dS = 0 $$

## Dynamic Hedging

In the Black-Scholes-Merton framework, perfect replication requires continuous rebalancing of the portfolio because $\Delta$ changes constantly with time ($t$) and asset price ($S$).

$$ \Delta_{BS} = N(d_1) $$

-   **Gamma Risk**: The rate of change of Delta ($\Gamma$) dictates how frequently the hedge needs adjustment. High Gamma near expiration (for ATM options) makes hedging difficult and expensive.
-   **Transaction Costs**: In practice, continuous rebalancing is impossible due to transaction costs. Traders rebalance periodically (e.g., daily or when Delta moves beyond a threshold).
