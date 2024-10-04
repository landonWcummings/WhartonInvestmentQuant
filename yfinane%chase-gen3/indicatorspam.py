import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
import random
import tqdm as tqdm
import pandas_ta as ta
import warnings
import shutil




def clean(data):
    split_columns = data["Date"].str.split("-", expand=True)
    
    split_columns.columns = ["Y","M","D"] #ik its not used here but can be used
    data = pd.concat([data, split_columns], axis=1)

    data['Date'] = pd.to_datetime(data['Date'])
    data['DayOfWeek'] = data['Date'].dt.day_name()
    day_mapping = {'Monday': 0,'Tuesday': 1,'Wednesday': 2,'Thursday': 3,'Friday': 4}
    data['DayOfWeek'] = data['DayOfWeek'].map(day_mapping)

    data["Y"] = pd.to_numeric(data["Y"], errors='coerce')
    data["M"] = pd.to_numeric(data["M"], errors='coerce')
    data["DayOfWeek"] = pd.to_numeric(data["DayOfWeek"], errors='coerce')

    finish_df = pd.DataFrame()
    if True: #MA
        MA = [7,10,15,25,50,100]
        for dist in MA:
            rolling_mean = data['lag_0'].rolling(window=dist, min_periods=1).mean()
            expanding_mean = data['lag_0'].expanding(min_periods=1).mean()
            finish_df[f'MA_{dist}'] = np.where(rolling_mean.isna(), expanding_mean, rolling_mean)
            finish_df[f'MA_{dist}'] = finish_df[f'MA_{dist}']/data['lag_0']
    if True: #RSI
        def calculate_rsi(row):
            lag_prices = row[['lag_0','lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 
                      'lag_6', 'lag_7', 'lag_8', 'lag_9', 'lag_10',
                      'lag_11', 'lag_12', 'lag_13', 'lag_14', 'lag_15']].values
    
            price_diff = np.diff(lag_prices)
            gains = np.where(price_diff > 0, price_diff, 0)
            losses = np.where(price_diff < 0, -price_diff, 0)
            average_gain = np.mean(gains)
            average_loss = np.mean(losses)
            
            if average_loss == 0:
                RSI = 100  
            else:
                RS = average_gain / average_loss
                RSI = 100 - (100 / (1 + RS))
            
            return RSI
        finish_df['RSI'] = data.apply(calculate_rsi, axis=1)
    if True: #MACD Signal_Line MACD_Hist
        macd = ta.macd(data['lag_0']).iloc[:, 0]
        finish_df['MACD'] = (macd - macd.mean()) / macd.std()

        signal_line = ta.macd(data['lag_0']).iloc[:, 1]
        finish_df['Signal_Line'] = (signal_line - signal_line.mean()) / signal_line.std()

        macd_hist = ta.macd(data['lag_0']).iloc[:, 2]
        finish_df['MACD_Hist'] = (macd_hist - macd_hist.mean()) / macd_hist.std()
    if True: #Bollinger Bands
        bollbands = [10,20]
        for boll in bollbands:
            bb = ta.bbands(data['lag_0'], length=boll)  # Returns a DataFrame
            upper_col = bb.columns[bb.columns.str.startswith(f'BBU_{boll}')][0]  # Upper band
            lower_col = bb.columns[bb.columns.str.startswith(f'BBL_{boll}')][0]  # Lower band
            bandwidth_col = bb.columns[bb.columns.str.startswith(f'BBB_{boll}')][0]  # Bandwidth
            percent_col = bb.columns[bb.columns.str.startswith(f'BBP_{boll}')][0]  # Percent
            
            # Assign to the DataFrame with custom column names
            finish_df[f'BB_up_{boll}'] = (bb[upper_col] / data['lag_0'] * 100)
            finish_df[f'BB_low_{boll}'] = (bb[lower_col] / data['lag_0'] * 100)
            finish_df[f'BB_bandwidth_{boll}'] = (bb[bandwidth_col] / data['lag_0'] * 100)
            finish_df[f'BB_percent_{boll}'] = bb[percent_col]
    if True: #Momentum
        momvals = [7,13]
        for val in momvals:
            finish_df[f'Momentum_{val}'] = (ta.mom(data['lag_0'], length=val) / data['lag_0'])
    if True: #CMO
        cmos = [12,17]
        for val in cmos:
            finish_df[f'CMO_{val}'] = (ta.cmo(data['lag_0'], length=val) / data['lag_0'])
    
    finaldata = finish_df.copy()
    lag =0
    for col in data.columns:
        if col.startswith('lag_'):
            lag += 1
    
    for j in range(lag-1):
        title = f'lag_{j+1}'
        titlen = f'lag_{j}'
        finaldata[title] = ((data[titlen]-data[title])/data[title]) * 100

    finaldata['targ'] = ((data['targ']-data['lag_0'])/data['lag_0']) * 100
    finaldata['Day'] = data['DayOfWeek']
    finaldata['M'] = data['M']
    finaldata['Y'] = data['Y']

    #clean up
    expel = finaldata[finaldata['MACD_Hist'].isna() | finaldata['Signal_Line'].isna() | finaldata['MACD'].isna()]
    finaldata = finaldata[~finaldata.index.isin(expel.index)]

    #split ----------
    split = finaldata[finaldata['Y'] <2020]
    olddata = finaldata[~finaldata.index.isin(split.index)]
    newdata = finaldata[finaldata.index.isin(split.index)]

    #olddata = olddata.drop(["Y"],axis=1)
    newdata = newdata.drop(["Y"],axis=1)

    nlag_columns = [col for col in newdata.columns if col.startswith('lag_')]
    olag_columns = [col for col in olddata.columns if col.startswith('lag_')]

    newdata = newdata.copy()
    newdata['zero_count'] = (newdata[nlag_columns] == 0).sum(axis=1)
    #newdata.loc[:, 'zero_count'] = (newdata[nlag_columns] == 0).sum(axis=1)
    newdata = newdata[newdata['zero_count'] <= 2]
    newdata = newdata.drop(columns=['zero_count'])

    olddata = olddata.copy()
    olddata['zero_count'] = (olddata[olag_columns] == 0).sum(axis=1)
    #olddata.loc[:, 'zero_count'] = (olddata[olag_columns] == 0).sum(axis=1)
    olddata = olddata[olddata['zero_count'] <= 2]
    olddata = olddata.drop(columns=['zero_count'])

    olddata = olddata.round(4)
    newdata = newdata.round(4)

    for col in olddata.columns:
            if olddata[col].dtype == 'float64':
                olddata[col] = olddata[col].astype('float32')
            if col == "Y":
                olddata = olddata.drop(columns=['Y'],axis=1)
    for col in newdata.columns:
            if newdata[col].dtype == 'float64':
                newdata[col] = newdata[col].astype('float32')
            if col == "Y":
                newdata = newdata.drop(columns=['Y'],axis=1)

    return olddata,newdata

def process_file(file_path,outputold, outputnew):
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        if not df.empty: 
            file_name = os.path.basename(file_path).replace('.txt', '')
            newdf, olddf = clean(df)  
            if olddf.size >= 2048:
                output_file_path = os.path.join(outputold, file_name)
                olddf.to_csv(output_file_path, index=False)
            if newdf.size >= 2048:
                output_file_path = os.path.join(outputnew, file_name)
                newdf.to_csv(output_file_path, index=False)
    else:
        os.remove(file_path)
if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import ProcessPoolExecutor

    outputold = r'C:\Users\lndnc\Downloads\YF\times\old'
    outputnew = r'C:\Users\lndnc\Downloads\YF\times\new'
    inputfolder = r'C:\Users\lndnc\Downloads\YF\classicprices'
    shutil.rmtree(outputnew); os.makedirs(outputnew)
    shutil.rmtree(outputold); os.makedirs(outputold)
    csv_files = glob.glob(os.path.join(inputfolder, '*'))



    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_file, file_path, outputold, outputnew) for file_path in csv_files]



"""data_frames = []
for file_path in csv_files:
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        if not df.empty:  
            file_name = os.path.basename(file_path)
            file_name = file_name.replace('.txt', '')
            newdf,olddf = clean(df)
            if olddf.size >= 2048:
                output_file_path = os.path.join(outputold, file_name)
                olddf.to_csv(output_file_path,index=False)
            if newdf.size >= 2048:
                output_file_path = os.path.join(outputnew, file_name)
                newdf.to_csv(output_file_path,index=False)
    else:
        os.remove(file_path)"""
