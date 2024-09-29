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

scoredatalocation = r"c:\Users\lndnc\Downloads\MAANGstockarchive\cleaneddata.csv"
scoredata = pd.read_csv(scoredatalocation)

csv_files = glob.glob(os.path.join(input_folder3, '*'))

i = 0
sums = 0


random.shuffle(csv_files)


data_frames = []
for file_path in csv_files:
    
    df = pd.read_csv(file_path)
    if not df.empty:  
        file_name = os.path.basename(file_path)
        file_name = file_name.replace('.csv.txt', '')
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
#params = {'learning_rate': 0.00241518328621528,'objective':'reg:squarederror', 'max_depth': 10, 'subsample': 0.6587886562063339, 'colsample_bytree': 0.6342255947815058, 'reg_alpha': 0.016728832792469464, 'reg_lambda': 0.0722322608658052, 'n_estimators': 278}
#etf params
#params =  {'learning_rate': 0.07875142331402334, 'max_depth': 18, 'subsample': 0.9393505448506804, 'colsample_bytree': 0.8134190497571698, 'reg_alpha': 0.0003866032766046731, 'reg_lambda': 0.0013868775148772124, 'n_estimators': 90}
#stock params
params = {'learning_rate': 0.01268628439424678, 'max_depth': 19, 'subsample': 0.8815261779053591, 'colsample_bytree': 0.9314936543240028, 'reg_alpha': 1.358163448010616, 'reg_lambda': 3.5931713252585333, 'n_estimators': 80} 
savepath = r"C:\Users\lndnc\Downloads\stockdataarchive\visdata.csv"
score = 0
goXGBmodel = XGB(X_train=X_train,X_test=X_test,y_test=y_test,y_train=y_train,
                savepath=savepath,groups=groups, optunadepth=2,params=params)
XGBmodel, score = goXGBmodel.makemodel()

print("-----------")
goXGBmodel.modscore(scoredatalocation,XGBmodel)
sums += score
i += 1
print(f'AVG differential after {i} iterations: {(sums / i).round(4)}')
print()

