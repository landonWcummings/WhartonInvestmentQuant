import pandas as pd
from sklearn.preprocessing import StandardScaler
#dataset from here
#  https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs/data

lag = 15
import pandas as pd
import os
import glob

# Set the folder paths
input_folder1 = r'C:\Users\lndnc\Downloads\stockdataarchive\Stocks'   
output_folder1 = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedStocks'

input_folder2 = r'C:\Users\lndnc\Downloads\stockdataarchive\ETFs'  
output_folder2 = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedETFs'


def modify(data):

    #split appart date column
    split_columns = data["Date"].str.split("-", expand=True)
    split_columns.columns = ["Y","a","b"] #ik its not used here but can be used
    data = pd.concat([data, split_columns], axis=1)

    #make year column numbers
    data["Y"] = pd.to_numeric(data["Y"], errors='coerce')

    #make lag number of lag rows
    for i in range(lag):
        data[f'lag_{i+1}'] = 0


    #fill lag columns with the close value that number of trading days before
    for lags in range(1, lag + 1): 
        data[f'lag_{lags}'] = data['Close'].shift(lags)

    #make a nextval column --------- This is the Y (what we train to predict)
    data[f'nextval'] = 0
    data[f'nextval'] = data['Close'].shift(-1)

    # drop all rows that have no lag values or no next val values (basically the first few and last 1 of every unique index)
    expel = data[data[f'lag_{lag}'].isna() | data[f'nextval'].isna()]
    data = data[~data.index.isin(expel.index)]

    #convert cols to category type for XGB
    for col in data.select_dtypes(include=['object']).columns:
        data.loc[:, col] = data[col].astype('category')    

    #drop not needed columns
    data = data.drop(["Date","Y","a","b","Volume","OpenInt"],axis=1)
    

    if False:
        data = data.drop(["Low","Open","High"],axis=1)

    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')

    return data



sums =0
count = 0
for file_path in glob.glob(os.path.join(input_folder1, '*')):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        count += 1
        sums += 1

        df = modify(df)

        file_name = os.path.basename(file_path).replace('.us', '.csv')
        output_file_path = os.path.join(output_folder1, file_name)

        df.to_csv(output_file_path, index=False)
        sums -= 1

for file_path in glob.glob(os.path.join(input_folder2, '*')):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        count += 1
        sums += 1

        df = modify(df)

        file_name = os.path.basename(file_path).replace('.us', '.csv')
        output_file_path = os.path.join(output_folder2, file_name)

        df.to_csv(output_file_path, index=False)
        sums -= 1


print(sums)
print(count)