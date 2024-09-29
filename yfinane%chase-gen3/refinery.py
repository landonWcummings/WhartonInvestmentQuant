import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import glob
import os
import random
import tqdm as tqdm


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
    #expel = data[data['Y'] <=2019]
    #data = data[~data.index.isin(expel.index)]
    data = data.drop(["Date","Y","D"],axis=1)

    lag_columns = [col for col in data.columns if col.startswith('lag_')]
    data['zero_count'] = (data[lag_columns] == 0).sum(axis=1)
    data = data[data['zero_count'] <= 2]
    data = data.drop(columns=['zero_count'])


    return data




base = r'C:\Users\lndnc\Downloads\YF\allrefined'
input = r'C:\Users\lndnc\Downloads\YF\classic'
csv_files = glob.glob(os.path.join(input, '*'))


data_frames = []
for file_path in csv_files:
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        if not df.empty:  
            file_name = os.path.basename(file_path)
            file_name = file_name.replace('.txt', '')
            df = clean(df)
            if df.size >= 2048:
                output_file_path = os.path.join(base, file_name)
                df.to_csv(output_file_path,index=False)
    else:
        os.remove(file_path)
