import numpy as np
import pandas as pd
from XGB import XGB
from sklearn.model_selection import train_test_split
import glob
import os
import random

evaldata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\cleanedtestdata.csv")
traindata = pd.read_csv(r"C:\Users\lndnc\Downloads\kagglestockdata\cleaneddata.csv")
#X_test = evaldata.drop("targ", axis=1)
#y_test = evaldata["targ"]
#X_train = traindata.drop("targ", axis=1)
#y_train = traindata["targ"]
del evaldata, traindata


#goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,optunadepth=10)
#XGBmodel = goXGBmodel.makemodel()


# Set the folder path where your CSV files are located
input_folder1 = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedETFs'
input_folder2 = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedStocks'

# Use glob to get all CSV file paths
csv_files = glob.glob(os.path.join(input_folder1, '*'))

i = 0
sums = 0
while i < 1000:
    random.shuffle(csv_files)


    data_frames = []
    for file_path in csv_files:
        
        df = pd.read_csv(file_path)
        if not df.empty:  
            file_name = os.path.basename(file_path)
            df.insert(0, 'tag', file_name)
            data_frames.append(df)
            
    combined_data = pd.concat(data_frames, ignore_index=True)
    X = combined_data.drop(["targ","tag"], axis=1)
    print(X.shape)
    y = combined_data["targ"]
    pregroups = combined_data["tag"]

    split_index = int(len(combined_data) * 0.85)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    groups = pregroups.iloc[split_index:]


    print("split - starting training")

    #params = {'learning_rate': 0.00241518328621528,'tree_method': 'hist','device':'cuda', 'max_depth': 20, 'subsample': 0.9587886562063339, 'colsample_bytree': 0.9342255947815058, 'reg_alpha': 0.016728832792469464, 'reg_lambda': 0.0722322608658052, 'n_estimators': 278}
    params = {'learning_rate': 0.00241518328621528,'objective':'reg:squarederror', 'max_depth': 50, 'subsample': 0.9587886562063339, 'colsample_bytree': 0.9342255947815058, 'reg_alpha': 0.016728832792469464, 'reg_lambda': 0.0722322608658052, 'n_estimators': 278}

    savepath = r"C:\Users\lndnc\Downloads\stockdataarchive\visdata.csv"
    score = 0
    goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,
                    savepath=savepath,groups=groups, optunadepth=20)
    XGBmodel, score = goXGBmodel.makemodel()
    sums += score
    i += 1
    print(f'AVG differential: {sums / i}')
print(f'Final avg differential: {sums / i}')
print()

