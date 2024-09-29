import numpy as np
import pandas as pd
import yfinance as yf
import tqdm as tqdm
lag = 15

def find(ticker):
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
        
        finaldata = pd.DataFrame()
        for j in range(lag):
            title = f'lag_{j+1}'
            titlen = f'lag_{j}'
            finaldata[title] = ((data[titlen]-data[title])/data[title]) * 100

        finaldata['targ'] = ((data['targ']-data['lag_0'])/data['lag_0']) * 100
        finaldata['Event'] = data['Event']
        finaldata['Date'] = data['Date']

        try:
            dividends = yf.Ticker(ticker).dividends
            dividend_dates = set(dividends.index.date)

            ticker_object = yf.Ticker(ticker)
            earnings_dates = ticker_object.earnings_dates

            if not earnings_dates.empty:
                earnings_dates_index = earnings_dates.index
                earnings_dates_list = pd.to_datetime(earnings_dates_index).date
                earnings_dates_set = set(earnings_dates_list)
            else:
                earnings_dates_set = set()

            # Create a single column for both dividends and earnings
            data.loc[data['Date'].dt.date.isin(dividend_dates), 'Event'] = 1 
            data.loc[data['Date'].dt.date.isin(earnings_dates_set), 'Event'] = 2  
        except Exception as e:
            print(f"Error retrieving special data for {ticker}: {e}")
        finaldata = finaldata.round(5)
    except Exception as e:
        print(f"Error retrieving data for {ticker}: {e}")
        return None
    return finaldata



listplace = r'c:\Users\lndnc\Downloads\iShares-Russell-3000-ETF_fund.csv'
df = pd.read_csv(listplace)
equity_tickers = df[df['Asset Class'] == 'Equity']['Ticker'].tolist()


base = r'C:\Users\lndnc\Downloads\YF\classic'
stocklist = ["aapl","meta"]
for ticker in tqdm.tqdm(equity_tickers, desc="Processing Tickers", unit="ticker"):

    data = find(ticker)
    if data is not None:

        save_path = f"{base}{r'\\'}{ticker}.csv"

        data.to_csv(save_path, index=False)



