import pandas as pd
#dataset from here
#  https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data/data?select=indexProcessed.csv
rawdata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\indexProcessed.csv")
output_folder1 = r"C:\Users\lndnc\Downloads\kagglestockdata\cleaneddata.csv"
savepath = r"C:\Users\lndnc\Downloads\kagglestockdata\cleanedtestdata.csv"
lag = 12
import os
import glob

if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)


def modify(data):
    
    split_columns = data["Date"].str.split("-", expand=True)
    split_columns.columns = ["Y","Month","day"] #ik its not used here but can be used
    data = pd.concat([data, split_columns], axis=1)

    #data['Date'] = pd.to_datetime(data['Date'])
    #data['DayOfWeek'] = data['Date'].dt.day_name()
    #day_mapping = {'Monday': 0,'Tuesday': 1,'Wednesday': 2,'Thursday': 3,'Friday': 4}
    #data['DayOfWeek'] = data['DayOfWeek'].map(day_mapping)

    #make year column numbers
    data["Month"] = pd.to_numeric(data["Month"], errors='coerce')

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
    data = data.drop(["Date","Y","Month","day","Volume","Adj Close"],axis=1)
    

    if True:
        data = data.drop(["Low","Open","High"],axis=1)

    
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
    finaldata = finaldata.round(5)


    return finaldata


grouped_data = rawdata.groupby('Index')
final_combined_data = pd.DataFrame()

for name, group in grouped_data:
    modified_group = modify(group)
    modified_group['Index'] = name  
    final_combined_data = pd.concat([final_combined_data, modified_group])

final_combined_data = final_combined_data.drop("Index", axis=1)
final_combined_data.to_csv(savepath, index=False)
