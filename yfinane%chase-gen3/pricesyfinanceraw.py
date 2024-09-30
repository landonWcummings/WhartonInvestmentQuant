import numpy as np
import pandas as pd
import yfinance as yf
import tqdm as tqdm
import concurrent.futures
import os


def find(ticker):
    lag = 15

    try:
        data = yf.download(ticker, period="max")
        data = data.reset_index()
        data['Event'] = 0
        
    
        for i in range(lag):
            data[f'lag_{i+1}'] = 0


        for lags in range(1, lag + 1): 
            data[f'lag_{lags}'] = data['Close'].shift(lags)

        data[f'targ'] = 0
        data[f'targ'] = data['Close'].shift(-1)

        expel = data[data[f'lag_{lag}'].isna() | data[f'targ'].isna()]
        data = data[~data.index.isin(expel.index)]

        for col in data.select_dtypes(include=['object']).columns:
            data.loc[:, col] = data[col].astype('category')    


        
        for col in data.columns:
            if data[col].dtype == 'float64':
                data[col] = data[col].astype('float32')

        data['lag_0'] = data['Close']
        data = data.drop("Close",axis=1)
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None
    
    return data




def save_ticker_data(ticker, base):
    data = find(ticker)
    if data is not None and data.size > 2048:
        save_path = os.path.join(base, f"{ticker}.csv")
        data.to_csv(save_path, index=False)

if __name__ == '__main__':
    lag = 15
    listplace = r'c:\Users\lndnc\Downloads\iShares-Russell-3000-ETF_fund.csv'
    df = pd.read_csv(listplace)
    equity_tickers = df[df['Asset Class'] == 'Equity']['Ticker'].tolist()

    base = r'C:\Users\lndnc\Downloads\YF\classicprices'
    
    # Use ProcessPoolExecutor for parallel processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Create a tqdm progress bar and map the tickers to the function
        list(tqdm.tqdm(executor.map(save_ticker_data, equity_tickers, [base] * len(equity_tickers)),
                        total=len(equity_tickers), desc="Processing Tickers", unit="ticker"))


