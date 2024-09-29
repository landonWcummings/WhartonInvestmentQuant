import pandas as pd
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
if not os.path.exists(output_folder1):
    os.makedirs(output_folder1)
if not os.path.exists(output_folder2):
    os.makedirs(output_folder2)


def modify(data):
    
    split_columns = data["Date"].str.split("-", expand=True)
    split_columns.columns = ["Y","M","D"] #ik its not used here but can be used
    data = pd.concat([data, split_columns], axis=1)

    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.day_name()
    day_mapping = {'Monday': 0,'Tuesday': 1,'Wednesday': 2,'Thursday': 3,'Friday': 4}
    data['DayOfWeek'] = data['DayOfWeek'].map(day_mapping)

    #make year column numbers
    data["M"] = pd.to_numeric(data["M"], errors='coerce')
    data["DayOfWeek"] = pd.to_numeric(data["DayOfWeek"], errors='coerce')

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
    
    


    
    for col in data.columns:
        if data[col].dtype == 'float64':
            data[col] = data[col].astype('float32')

    data['lag_0'] = data['Close']
    data = data.drop("Close",axis=1)
    
    lag_columns = [col for col in data.columns if col.startswith('lag_')]
    data['zero_count'] = (data[lag_columns] == 0).sum(axis=1)
    data = data[data['zero_count'] <= 2]
    data = data.drop(columns=['zero_count'])


    finaldata = pd.DataFrame()
    for j in range(lag):
        title = f'lag_{j+1}'
        titlen = f'lag_{j}'
        finaldata[title] = ((data[titlen]-data[title])/data[title]) * 100

    finaldata['targ'] = ((data['targ']-data['lag_0'])/data['lag_0']) * 100
    finaldata['M'] = data['M']
    finaldata['DayOfWeek'] = data['DayOfWeek']



    finaldata = finaldata.round(5)


    return finaldata



sums =0
count = 0


for file_path in glob.glob(os.path.join(input_folder1, '*')):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        count += 1
        sums += 1

        df = modify(df)
        if df.size > 2048:
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