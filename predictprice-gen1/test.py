import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\lndnc\Downloads\stockdataarchive\visdata.csv")

PL_base = ((data["Price Change"] / data['Close']) * 100).sum()
PL_sum = data["PL"].sum()
print(f'Base Profit/Loss : {PL_base}')
print(f'Profit/Loss : {PL_sum}')
