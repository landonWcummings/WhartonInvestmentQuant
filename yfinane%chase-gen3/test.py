import yfinance as yf
import pandas as pd

# Example: Get data for Apple (AAPL)
ticker = yf.Ticker("AAPL")

# Current PE ratio
pe_ratio = ticker.info.get('trailingPE')
print(f"Current PE Ratio: {pe_ratio}")

# Revenue growth
quarterly_financials = ticker.quarterly_financials
#revenue_growth = quarterly_financials.loc['Total Revenue'].pct_change()
#print("Revenue Growth:\n", revenue_growth)

# Profit margin
profit_margin = ticker.info.get('profitMargins', 'N/A')
print(f"Profit Margin: {profit_margin}")

# Sector
sector = ticker.info.get('sector', 'N/A')
print(f"Sector: {sector}")

quarterly_financials.to_csv(r'C:\Users\lndnc\Downloads\YF\lookaapl.csv',index=True)
