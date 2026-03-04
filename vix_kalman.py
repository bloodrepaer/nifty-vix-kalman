import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.stats import norm

def get_market_data():
    symbols = ['^NSEI', '^INDIAVIX']
    prices_df = yf.download(symbols, start='2018-01-01', end='2024-01-01', progress=False)['Close']
    
    # forward-fill handles non-trading days to maintain state continuity. 
    # note: spot index ignores dividend drag, acceptable for gross directional signals.
    return prices_df.ffill().dropna()

def estimate_vol_state(vix_series):
    # volatility clustering assumption: state 1 models the latent momentum of iv expansion.
    # transition covariance is tuned low to avoid over-fitting intraday market noise.
    vol_filter = KalmanFilter(
        transition_matrices=[[1, 1], [0, 1]],
        observation_matrices=[[1, 0]],
        initial_state_mean=[vix_series.iloc[0], 0],
        initial_state_covariance=np.eye(2),
        transition_covariance=np.eye(2) * 1e-4,
        observation_covariance=1e-2
    )
    
    filtered_states, _ = vol_filter.filter(vix_series.values)
    
    state_df = pd.DataFrame(index=vix_series.index)
    state_df['VIX_Level'] = filtered_states[:, 0]
    state_df['VIX_Velocity'] = filtered_states[:, 1]
    
    return state_df

def run_backtest(prices_df, vol_states):
    nifty_returns = prices_df['^NSEI'].pct_change().fillna(0)
    
    # regime-switching sizing: de-grossing to 20% delta exposure during 
    # volatility expansion (fear regime) to mitigate left-tail risk.
    signal_vector = np.where(vol_states['VIX_Velocity'] > 0, 0.2, 1.0)
    
    # t+1 execution: rolling target weights by 1 to simulate trading at t+1 open.
    # assumes zero market impact and instantaneous liquidity.
    position_array = np.roll(signal_vector, 1)
    position_array[0] = 0 
    
    # fixed slippage model: 10 bps per leg. 
    # a volume-weighted impact model (e.g., almgren-chriss) would be needed for larger aum.
    turnover = np.abs(np.diff(position_array, prepend=0))
    slippage_costs = turnover * 0.001 
    
    portfolio_returns = (position_array * nifty_returns) - slippage_costs
    
    portfolio_equity = 100000 * np.exp(np.cumsum(portfolio_returns))
    benchmark_equity = 100000 * np.exp(np.cumsum(nifty_returns))
    
    return portfolio_returns, portfolio_equity, benchmark_equity

def print_metrics(portfolio_returns, portfolio_equity, benchmark_equity):
    # note: sharpe calculation assumes a 0% risk-free rate for simplicity.
    ann_ret = np.mean(portfolio_returns) * 252
    ann_vol = np.std(portfolio_returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    
    peak = np.maximum.accumulate(portfolio_equity)
    max_dd = np.min((portfolio_equity - peak) / peak)
    
    # historical var used over parametric due to the leptokurtic (fat-tailed) 
    # nature of equity returns, especially during structural breaks.
    var_hist = np.percentile(portfolio_returns, 1)
    
    print("\n--- Strategy Results ---")
    print(f"Annualized Return: {ann_ret*100:.2f}%")
    print(f"Annualized Vol:    {ann_vol*100:.2f}%")
    print(f"Sharpe Ratio:      {sharpe:.2f}")
    print(f"Max Drawdown:      {max_dd*100:.2f}%")
    print(f"99% Daily VaR:     {var_hist*100:.2f}%")

if __name__ == "__main__":
    market_data = get_market_data()
    vol_states = estimate_vol_state(market_data['^INDIAVIX'])
    p_rets, p_equity, b_equity = run_backtest(market_data, vol_states)
    print_metrics(p_rets, p_equity, b_equity)