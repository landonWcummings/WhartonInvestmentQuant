import numpy as np
import pandas as pd
from XGB import XGB
from sklearn.model_selection import train_test_split
import glob
import os
import random

evaldata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\cleanedtestdata.csv")
traindata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\cleaneddata.csv")
#X_test = evaldata.drop("nextval", axis=1)
#y_test = evaldata["nextval"]
#X_train = traindata.drop("nextval", axis=1)
#y_train = traindata["nextval"]
del evaldata, traindata


#goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,optunadepth=10)
#XGBmodel = goXGBmodel.makemodel()


# Set the folder path where your CSV files are located
input_folder = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedETFs'

# Use glob to get all CSV file paths
csv_files = glob.glob(os.path.join(input_folder, '*'))
random.shuffle(csv_files)


data_frames = []
base_PL = 0
for file_path in csv_files:
    
    df = pd.read_csv(file_path)
    if not df.empty and 'Close' in df.columns:  
        file_name = os.path.basename(file_path)
        df.insert(0, 'ETF', file_name)
        data_frames.append(df)
        
combined_data = pd.concat(data_frames, ignore_index=True)
X = combined_data.drop(["nextval","ETF"], axis=1)
print(X.shape)
y = combined_data["nextval"]
pregroups = combined_data["ETF"]

split_index = int(len(combined_data) * 0.85)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]
groups = pregroups.iloc[split_index:]


print("split - starting training")

params = {
                'learning_rate': 0.003602747587780287,  
                'max_depth': 200,
                'subsample': 0.6430592894287979,  
                'colsample_bytree': 0.6960187402094009, 
                'reg_alpha': 5.157320016202913, 
                'reg_lambda': 3.326691797391068, 
                'n_estimators': 550,
                'enable_categorical': False
            }

savepath = r"C:\Users\lndnc\Downloads\stockdataarchive\visdata.csv"
goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,
                savepath=savepath,groups=groups, optunadepth=300,params=params)
XGBmodel = goXGBmodel.makemodel()

