#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Importing Required Libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the path to the folder containing your CSV files
folder_path = "C:/Users/tejas/Downloads/Historic stock prices/historical_data"

# Initialize an empty list to store data
data_frames = []

# Loop through all CSV files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):  # Only process CSV files
        file_path = os.path.join(folder_path, file_name)
        # Read the CSV file
        stock_data = pd.read_csv(file_path)
        # Ensure it has 'Date' and 'Close' columns (adjust if needed)
        stock_data = stock_data[['Date', 'Close']].copy()
        # Rename the 'Close' column to include the stock name (from the file name)
        stock_name = os.path.splitext(file_name)[0]  # Remove file extension
        stock_data.rename(columns={'Close': f'{stock_name}_Close'}, inplace=True)
        # Append to the list
        data_frames.append(stock_data)

# Merge all data frames on the 'Date' column
combined_data = data_frames[0]
for df in data_frames[1:]:
    combined_data = pd.merge(combined_data, df, on='Date', how='outer')

# Sort by Date and fill any missing data
combined_data['Date'] = pd.to_datetime(combined_data['Date'])
combined_data.sort_values(by='Date', inplace=True)
combined_data.fillna(method='ffill', inplace=True)

# Calculate daily returns for each stock
for column in combined_data.columns:
    if column.endswith('_Close'):
        stock_name = column.split('_')[0]
        combined_data[f'{stock_name}_Returns'] = combined_data[column].pct_change()

# Drop the first row with NaN values
combined_data.dropna(inplace=True)

# Calculate Portfolio Returns
weights = [0.4, 0.3, 0.3]  # Adjust weights based on your portfolio allocation
returns_columns = [col for col in combined_data.columns if col.endswith('_Returns')]
combined_data['Portfolio_Returns'] = sum(weight * combined_data[col] for weight, col in zip(weights, returns_columns))

# Visualize Portfolio Returns Distribution
sns.histplot(combined_data['Portfolio_Returns'], kde=True, bins=30, color='blue', alpha=0.6)
plt.title('Portfolio Returns Distribution')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.show()

# Monte Carlo Simulation
simulations = 10000
simulated_portfolio_returns = np.random.choice(combined_data['Portfolio_Returns'], size=(simulations, 1), replace=True)

# Calculating VaR and Expected Shortfall using Monte Carlo
confidence_level = 0.95
VaR_mc = np.percentile(simulated_portfolio_returns, (1 - confidence_level) * 100)
ES_mc = simulated_portfolio_returns[simulated_portfolio_returns <= VaR_mc].mean()

print(f"Monte Carlo 95% Value at Risk (VaR): {VaR_mc:.4f}")
print(f"Monte Carlo 95% Expected Shortfall (ES): {ES_mc:.4f}")

# Visualization of Monte Carlo VaR and ES
plt.figure(figsize=(10, 6))
sns.histplot(simulated_portfolio_returns, kde=True, bins=50, color='blue', alpha=0.6)
plt.axvline(x=VaR_mc, color='red', linestyle='--', label=f'VaR (95%): {VaR_mc:.4f}')
plt.axvline(x=ES_mc, color='orange', linestyle='--', label=f'ES (95%): {ES_mc:.4f}')
plt.title('Monte Carlo Risk Metrics')
plt.xlabel('Simulated Returns')
plt.ylabel('Frequency')
plt.legend()
plt.show()

# Save Results
summary = pd.DataFrame({
    'Metric': ['Monte Carlo Value at Risk (VaR)', 'Monte Carlo Expected Shortfall (ES)'],
    'Value': [VaR_mc, ES_mc]
})
summary.to_csv('monte_carlo_financial_risk_summary.csv', index=False)
print(summary)


# In[ ]:




