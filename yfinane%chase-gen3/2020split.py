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

    expel = data[data['Y'] <2022]
    olddata = data[~data.index.isin(expel.index)]
    newdata = data[data.index.isin(expel.index)]

    olddata = olddata.drop(["Date","Y","D"],axis=1)
    newdata = newdata.drop(["Date","Y","D"],axis=1)

    nlag_columns = [col for col in newdata.columns if col.startswith('lag_')]
    olag_columns = [col for col in olddata.columns if col.startswith('lag_')]

    newdata['zero_count'] = (newdata[nlag_columns] == 0).sum(axis=1)
    newdata = newdata[newdata['zero_count'] <= 2]
    newdata = newdata.drop(columns=['zero_count'])

    olddata['zero_count'] = (olddata[olag_columns] == 0).sum(axis=1)
    olddata = olddata[olddata['zero_count'] <= 2]
    olddata = olddata.drop(columns=['zero_count'])


    return olddata,newdata




outputold = r'c:\Users\lndnc\Downloads\YF\timesplit\new'
outputnew = r'C:\Users\lndnc\Downloads\YF\timesplit\old'
input = r'C:\Users\lndnc\Downloads\YF\classic'
csv_files = glob.glob(os.path.join(input, '*'))


data_frames = []
for file_path in csv_files:
    if os.path.getsize(file_path) > 2048:
        df = pd.read_csv(file_path)
        if not df.empty:  
            file_name = os.path.basename(file_path)
            file_name = file_name.replace('.txt', '')
            olddf,newdf = clean(df)
            if olddf.size >= 2048:
                output_file_path = os.path.join(outputold, file_name)
                olddf.to_csv(output_file_path,index=False)
            if newdf.size >= 2048:
                output_file_path = os.path.join(outputnew, file_name)
                newdf.to_csv(output_file_path,index=False)
    else:
        os.remove(file_path)
