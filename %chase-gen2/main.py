import numpy as np
import pandas as pd
from XGBp import XGB
from sklearn.model_selection import train_test_split
import glob
import os
import random



#goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,optunadepth=10)
#XGBmodel = goXGBmodel.makemodel()


input_folder1 = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedETFs'
input_folder2 = r'C:\Users\lndnc\Downloads\stockdataarchive\cleanedStocks'
input_folder3 = r'C:\Users\lndnc\Downloads\modernstockdata\cleaneddata'
input_folder4 = r'C:\Users\lndnc\Downloads\YF\allrefined'
input_folder5 = r'C:\Users\lndnc\Downloads\YF\timesplit\old'
input_folder6 = r'C:\Users\lndnc\Downloads\YF\times\old'


csv_files = glob.glob(os.path.join(input_folder6, '*'))

i = 0
sums = 0
sums2 = 0

while i < 1000:
    random.shuffle(csv_files)


    data_frames = []
    for file_path in csv_files:
        
        df = pd.read_csv(file_path)
        if not df.empty:  
            #df = df.drop("Event",axis=1)
            file_name = os.path.basename(file_path)
            file_name = file_name.replace('.csv.txt', '')
            df.insert(0, 'tag', file_name)
            data_frames.append(df)
            
    combined_data = pd.concat(data_frames, ignore_index=True)
    X = combined_data.drop(["targ","tag"], axis=1)
    print(X.shape)
    y = combined_data["targ"]
    pregroups = combined_data["tag"]

    split_index = int(len(combined_data) * 0.91)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    groups = pregroups.iloc[split_index:]


    print("split - starting training")

    #params = {'learning_rate': 0.00241518328621528,'tree_method': 'hist','device':'cuda', 'max_depth': 20, 'subsample': 0.9587886562063339, 'colsample_bytree': 0.9342255947815058, 'reg_alpha': 0.016728832792469464, 'reg_lambda': 0.0722322608658052, 'n_estimators': 278}
    #params = {'learning_rate': 0.00241518328621528,'objective':'reg:squarederror', 'max_depth': 10, 'subsample': 0.6587886562063339, 'colsample_bytree': 0.6342255947815058, 'reg_alpha': 0.016728832792469464, 'reg_lambda': 0.0722322608658052, 'n_estimators': 278}
    #etf params
    #params =  {'learning_rate': 0.07875142331402334, 'max_depth': 18, 'subsample': 0.9393505448506804, 'colsample_bytree': 0.8134190497571698, 'reg_alpha': 0.0003866032766046731, 'reg_lambda': 0.0013868775148772124, 'n_estimators': 90}
    #stock params
    
    params =   {'learning_rate': 0.051288347144786264, 'max_depth': 10, 'subsample': 0.8826703824895863, 'colsample_bytree': 0.9904704822382722, 'reg_alpha': 0.5418850241691432, 'reg_lambda': 0.2887563399981681, 'n_estimators': 204}
    params['n_estimators'] = params['n_estimators'] +i
    params['max_depth'] = params['max_depth'] + i//6
    savepath = r"C:\Users\lndnc\Downloads\stockdataarchive\visdata.csv"
    #scoredatalocation = r"c:\Users\lndnc\Downloads\stockdataarchive\allstocks.csv"
    scoredatalocation = r"c:\Users\lndnc\Downloads\YF\times\allnew.csv"
    score = 0
    goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,
                    savepath=savepath,groups=groups, 
                    optunadepth=12,gpu=True,params=params)
    XGBmodel, score = goXGBmodel.makemodel()
    score2 = goXGBmodel.modscore(scoredatalocation,XGBmodel,iterations=i)
    sums += score
    sums2 += score2
    i += 1
    print(f'AVG differential after {i} iterations: {(sums / i).round(4)}')
    print(f'AVG test differential after {i} iterations: {(sums2 / i).round(4)}')
    print(params)

print(f'Final avg differential: {sums / i}')
print()

