import pandas as pd
import numpy as np
#dataset from here
#  https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data/data?select=indexProcessed.csv
inputfolder = r"C:\Users\lndnc\Downloads\modernstockdata\Daily\Daily"

output_folder1 = r"C:\Users\lndnc\Downloads\modernstockdata\cleaneddata"
lag = 15
import os
import glob

if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)


def modify(data):
    
    #split_columns = data["Date"].str.split("-", expand=True)
    #split_columns.columns = ["Y","Month","day"] #ik its not used here but can be used
    #data = pd.concat([data, split_columns], axis=1)
    #data["Y"] = pd.to_numeric(data["Y"], errors='coerce')
    #expel = data[data['Y'] <=2021]
    #data = data[~data.index.isin(expel.index)]

    #data['Date'] = pd.to_datetime(data['Date'])
    #data['DayOfWeek'] = data['Date'].dt.day_name()
    #day_mapping = {'Monday': 0,'Tuesday': 1,'Wednesday': 2,'Thursday': 3,'Friday': 4}
    #data['DayOfWeek'] = data['DayOfWeek'].map(day_mapping)

    #make year column numbers
    #data["Month"] = pd.to_numeric(data["Month"], errors='coerce')

    #make lag number of lag rows
    for i in range(lag):
        data[f'lag_{i+1}'] = 0


    #fill lag columns with the close value that number of trading days before
    for lags in range(1, lag + 1): 
        data[f'lag_{lags}'] = data['Close'].shift(lags)

    #make a targ column --------- This is the Y (what we train to predict)
    data[f'targ'] = 0
    data[f'targ'] = data['Close'].shift(-1)

    # drop all rows that have no lag values or no next val values (basically the first few and last 1 of every unique index)
    expel = data[data[f'lag_{lag}'].isna() | data[f'targ'].isna()]
    data = data[~data.index.isin(expel.index)]

    #convert cols to category type for XGB
    for col in data.select_dtypes(include=['object']).columns:
        data.loc[:, col] = data[col].astype('category')    

    #drop not needed columns
    data = data.drop(["Volume","Adj Close"],axis=1)
    

    if True:
        data = data.drop(["Low","Open","High"],axis=1)

    
    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')

    data['lag_0'] = data['Close']
    data = data.drop("Close",axis=1)

    lag_columns = [col for col in data.columns if 'lag' in col]
    condition1 = (data[lag_columns] == 0).sum(axis=1) <= 2

    condition2 = data[lag_columns].abs().median(axis=1) < 5
    data = data[condition1 & condition2]



    finaldata = pd.DataFrame()
    for j in range(lag):
        title = f'lag_{j+1}'
        titlen = f'lag_{j}'
        finaldata[title] = ((data[titlen]-data[title])/data[title]) * 100

    finaldata['targ'] = ((data['targ']-data['lag_0'])/data['lag_0']) * 100
    finaldata = finaldata.replace([np.inf, -np.inf], np.nan).dropna()
    finaldata = finaldata.round(5)


    return finaldata


final_combined_data = pd.DataFrame()
for file_path in glob.glob(os.path.join(inputfolder, '*')):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_parquet(file_path)
        df = modify(df)
        file_name = os.path.basename(file_path).replace('.parquet', '.csv')
        output_file_path = os.path.join(output_folder1, file_name)

        df.to_csv(output_file_path, index=False)

