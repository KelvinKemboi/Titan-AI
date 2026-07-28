# The "Brain" Weights - Tuned for current market conditions (High Volatility)
# Hidden from the user to maintain "Black Box" simplicity
WEIGHTS = {
    'Value': 0.25, # Undervalued stocks
    'Momentum': 0.30, # Stocks already moving up
    'Quality': 0.25, # High margins / Profitable
    'Solvency': 0.10, # Low Debt
    'Volatility': 0.10, # Low Beta (Risk)
}
