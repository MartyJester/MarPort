import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

print("ciao")


# List of ETFs (replace with your choices)
etfs = ['SPY', 'QQQ', 'EFA', 'TLT', 'GLD']

# Download data
df = yf.download(etfs, start="2010-01-01", end="2024-03-10")["Close"]

# Save to CSV (optional)
df.to_csv('etf_prices.csv')


# Load historical data (assume df has Date as index and ETFs as columns)
df = pd.read_csv('etf_prices.csv', index_col='Date', parse_dates=True)

# Compute log returns
returns = np.log(df / df.shift(1)).dropna()

# Expected returns (mean of historical returns)
expected_returns = returns.mean()

# Covariance matrix
cov_matrix = returns.cov()

# Define objective function (minimize portfolio variance)
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)

# Constraints: Weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

# Bounds: No short selling (0 <= w <= 1)
bounds = [(0, 1)] * len(expected_returns)

# Initial weights (equal allocation)
initial_weights = np.ones(len(expected_returns)) / len(expected_returns)

# Minimize portfolio variance (risk)
opt_result = minimize(portfolio_volatility, initial_weights, args=(cov_matrix,),
                      method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights
optimal_weights = opt_result.x

# Portfolio expected return & risk
optimal_return = np.dot(optimal_weights, expected_returns)
optimal_risk = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)

print("Optimal Weights:", optimal_weights)
print("Expected Return:", optimal_return)
print("Risk (Volatility):", optimal_risk)


num_portfolios = 10000
weights_record = []
returns_record = []
risks_record = []

for _ in range(num_portfolios):
    weights = np.random.dirichlet(np.ones(len(expected_returns)), size=1)[0]  # Sum to 1
    weights_record.append(weights)
    returns_record.append(np.dot(weights, expected_returns))
    risks_record.append(np.sqrt(weights.T @ cov_matrix @ weights))

# Convert to NumPy arrays
returns_record = np.array(returns_record)
risks_record = np.array(risks_record)

# Plot Efficient Frontier
plt.figure(figsize=(10, 6))
plt.scatter(risks_record, returns_record, c=returns_record/risks_record, cmap='viridis', alpha=0.5)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Risk (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier")
plt.scatter(optimal_risk, optimal_return, color='red', marker='*', s=200, label="Optimal Portfolio")
plt.legend()
plt.show()
