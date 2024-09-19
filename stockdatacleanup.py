import pandas as pd
#  https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data/data?select=indexProcessed.csv
rawdata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\indexProcessed.csv")
savepath1 = r"C:\Users\lndnc\Downloads\kagglestockdata\cleaneddata.csv"
savepath = r"C:\Users\lndnc\Downloads\kagglestockdata\inspectcleaneddata.csv"
lag = 12


for name in rawdata.columns:
    null_count = rawdata[name].isnull().sum()
    
    print(name + " - " + str(null_count))


split_columns = rawdata["Date"].str.split("-", expand=True)
split_columns.columns = ["Y","a","b"]

rawdata = pd.concat([rawdata, split_columns], axis=1)

rawdata["Y"] = pd.to_numeric(rawdata["Y"], errors='coerce')
evaldata = rawdata[rawdata["Y"] > 2018]
rawdata = rawdata[~rawdata.index.isin(evaldata.index)]


rawdata = rawdata.drop(["CloseUSD","Date","Y","a","b"],axis=1)
evaldata = evaldata.drop(["CloseUSD","Date","Y","a","b"],axis=1)

for i in range(lag):
    rawdata[f'lag_{i+1}'] = 0

grouped_data = rawdata.groupby('Index')

for lags in range(1, lag + 1): 
    rawdata[f'lag_{lags}'] = grouped_data['Close'].shift(lags)



print(rawdata.columns)
print(rawdata.head())
print(rawdata.dtypes)
print("________")

rawdata.to_csv(savepath1, index=False)
evaldata.to_csv(savepath, index=False)