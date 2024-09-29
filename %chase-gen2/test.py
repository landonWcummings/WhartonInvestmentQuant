import numpy as np
import pandas as pd
data = pd.read_csv(r"C:\Users\lndnc\Downloads\stockdataarchive\visdata.csv")


#data['Direction predicted'] = True
#data['is direction correct'] = (data['actual direction'] == data['Direction predicted'])

direction_counts = data['is direction correct'].value_counts()

data['PL'] = np.where(data['is direction correct'], data['basePL'].abs(), -1 * data['basePL'].abs())

print(direction_counts)

print(f'Biggest P: {data["PL"].max()}')
print(f'Biggest L: {data["PL"].min()}')

long_trades = data.loc[data['Direction predicted'] == True].copy()
short_trades = data.loc[data['Direction predicted'] == False].copy()

# Calculate PL for long trades
long_trades['PL'] = long_trades['Actual']

# Calculate PL for short trades
short_trades['PL'] = -short_trades['Actual']

long_PL_sum = long_trades['PL'].sum()
short_PL_sum = short_trades['PL'].sum()

print(f'Long Positions Profit/Loss: ${long_PL_sum:.2f}')
print(f'Short Positions Profit/Loss: ${short_PL_sum:.2f}')



PL_base = (data['basePL']).sum()
PL_sum = data["PL"].sum()
print(f'Base Profit/Loss : {PL_base}')
print(f'Profit/Loss: {PL_sum}')
print(f'Market differential: {PL_sum - PL_base}')