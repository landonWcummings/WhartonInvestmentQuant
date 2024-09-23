import pandas as pd
from sklearn.preprocessing import StandardScaler
#dataset from here
#  https://www.kaggle.com/datasets/mattiuzc/stock-exchange-data/data?select=indexProcessed.csv
rawdata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\indexProcessed.csv")
savepath1 = r"C:\Users\lndnc\Downloads\kagglestockdata\cleaneddata.csv"
savepath = r"C:\Users\lndnc\Downloads\kagglestockdata\cleanedtestdata.csv"
lag = 12

#see if we have any NAN
for name in rawdata.columns:
    null_count = rawdata[name].isnull().sum()
    print(name + " - " + str(null_count))

#split appart date column
split_columns = rawdata["Date"].str.split("-", expand=True)
split_columns.columns = ["Y","a","b"]
rawdata = pd.concat([rawdata, split_columns], axis=1)

#make year column numbers
rawdata["Y"] = pd.to_numeric(rawdata["Y"], errors='coerce')

#make lag number of lag rows
for i in range(lag):
    rawdata[f'lag_{i+1}'] = 0

#groups based on Index
grouped_data = rawdata.groupby('Index')

#fill lag columns with the close value that number of trading days before
for lags in range(1, lag + 1): 
    rawdata[f'lag_{lags}'] = grouped_data['Close'].shift(lags)

#make a nextval column --------- This is the Y (what we train to predict)
rawdata[f'nextval'] = 0
rawdata[f'nextval'] = grouped_data['Close'].shift(-1)

# drop all rows that have no lag values or no next val values (basically the first few and last 1 of every unique index)
expel = rawdata[rawdata[f'lag_{lag}'].isna() | rawdata[f'nextval'].isna()]
rawdata = rawdata[~rawdata.index.isin(expel.index)]

#convert cols to category type for XGB
for col in rawdata.select_dtypes(include=['object']).columns:
    rawdata.loc[:, col] = rawdata[col].astype('category')


#seperate a evaluation set to test the model
evaldata = rawdata[rawdata["Y"] > 2018]
rawdata = rawdata[~rawdata.index.isin(evaldata.index)]

#drop not needed columns
rawdata = rawdata.drop(["CloseUSD","Date","Y","a","b","Adj Close","Volume","Index"],axis=1)
evaldata = evaldata.drop(["CloseUSD","Date","Y","a","b","Adj Close","Volume","Index"],axis=1)

if True:
    rawdata = rawdata.drop(["Low","Open","High"],axis=1)
    evaldata = evaldata.drop(["Low","Open","High"],axis=1)


print(rawdata.dtypes)

print(rawdata.columns)
print(rawdata.head())
print(rawdata.dtypes)
print("________")

rawdata.to_csv(savepath1, index=False)
evaldata.to_csv(savepath, index=False)