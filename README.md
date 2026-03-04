# Vectorized NIFTY Volatility Strategy Engine

A quantitative research pipeline that applies a state-space model (Kalman Filter) to the India VIX to track latent volatility momentum and dynamically adjust delta exposure on the NIFTY 50 index. 

Built with a focus on fast iteration cycles, pure NumPy vectorization, and rigorous risk management.

## Mathematical Intuition
Options pricing and market regimes are heavily driven by volatility clustering. Standard moving averages often lag or overreact to intraday noise. 
To extract the true underlying volatility trend, I implemented a **Kalman Filter**. By treating the India VIX as a noisy physical system, the filter estimates two hidden states:
1. **Level:** The true underlying volatility.
2. **Velocity:** The momentum/trend of the volatility expansion or contraction.

When the filter detects positive volatility velocity (a transition into a high-fear regime), the engine systematically de-grosses NIFTY delta exposure to mitigate left-tail risk.

## Technical Architecture
* **Pure Vectorization:** The backtest engine is built entirely without iterative `for` loops. Signal generation, position sizing, and equity curve calculations are executed using element-wise NumPy array operations (`np.where`, `np.cumsum`, `np.diff`) for optimal performance on large datasets.
* **Look-Ahead Bias Prevention:** Target weights are explicitly rolled forward by one period (`np.roll`) to simulate $T+1$ execution, ensuring signals generated at the close are traded at the next open.
* **Friction Modeling:** Incorporates a fixed transaction cost model (10 bps) calculated on absolute portfolio turnover.

## Risk & Performance Metrics
The engine calculates standard institutional risk metrics, acknowledging the leptokurtic (fat-tailed) nature of equity returns:
* Annualized Return & Volatility
* Sharpe Ratio (assuming 0% risk-free rate)
* Maximum Drawdown
* 99% Value at Risk (VaR) using Historical Simulation

## Dependencies
* `numpy`
* `pandas`
* `yfinance`
* `pykalman`
* `scipy`
* `matplotlib`

## How to Run
```bash
git clone [https://github.com/yourusername/vix-kalman-engine.git](https://github.com/yourusername/vix-kalman-engine.git)
cd vix-kalman-engine
pip install -r requirements.txt
python vix_kalman.py
