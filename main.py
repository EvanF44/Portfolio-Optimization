import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --- Data ---
tickers = ['AAPL', 'MSFT', 'JPM', 'VOO', 'IEF']
data = yf.download(tickers, start='2019-01-01', end='2024-01-01')['Close']
data = data[tickers]  # yfinance returns columns alphabetically, so reorder to match tickers
returns = data.pct_change().dropna()

# --- Inputs ---
num_portfolios = 10000
risk_free_rate = 0.04  # assumed annual risk-free rate
mean_returns = returns.mean()
cov_matrix = returns.cov()

results = np.zeros((3, num_portfolios))  # rows: return, volatility, Sharpe
weights_record = []

# --- Monte Carlo simulation ---
np.random.seed(42)
for i in range(num_portfolios):
    weights = np.random.random(len(tickers))
    weights /= np.sum(weights)  # weights sum to 1, long-only
    weights_record.append(weights)

    # annualize with 252 trading days
    p_return = np.sum(mean_returns * weights) * 252
    p_volatility = np.sqrt(weights @ (cov_matrix * 252) @ weights)
    p_sharpe = (p_return - risk_free_rate) / p_volatility

    results[0, i] = p_return
    results[1, i] = p_volatility
    results[2, i] = p_sharpe

# --- Key portfolios ---
max_sharpe_idx = np.argmax(results[2])
min_vol_idx = np.argmin(results[1])


def print_portfolio(name, idx):
    print(f"=== {name} ===")
    print(f"Annualized Return:     {results[0, idx]:.2%}")
    print(f"Annualized Volatility: {results[1, idx]:.2%}")
    print(f"Sharpe Ratio:          {results[2, idx]:.2f}")
    for ticker, weight in zip(tickers, weights_record[idx]):
        print(f"  {ticker}: {weight:.2%}")
    print()


print_portfolio("MAX SHARPE PORTFOLIO", max_sharpe_idx)
print_portfolio("MIN VOLATILITY PORTFOLIO", min_vol_idx)

# --- Efficient frontier plot ---
plt.figure(figsize=(10, 6))
scatter = plt.scatter(results[1], results[0], c=results[2], cmap='viridis', alpha=0.5, s=10)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.scatter(results[1, max_sharpe_idx], results[0, max_sharpe_idx],
            color='red', s=150, marker='*', label='Max Sharpe')
plt.scatter(results[1, min_vol_idx], results[0, min_vol_idx],
            color='blue', s=100, marker='v', label='Min Volatility')

plt.title('Monte Carlo Efficient Frontier (2019-2023)')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('efficient_frontier.png', dpi=150, bbox_inches='tight')
plt.show()