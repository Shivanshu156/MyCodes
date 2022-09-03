import pandas as pd
import os

filename = 'data_clean_added_C.csv'
output = 'element_wise_features_2.csv'

data = pd.read_csv(filename)

# data = data.loc[:,'Element 1':'SE']
data.drop(columns=['Ratio','Element 2','Carbon B.E.','Oxygen B.E.'], inplace=True)
col  = []
for column in data.columns:
    col.append(column)

for i in range(len(col)):
    print( col[i], i)
del col[13:17]
col
data = data[col]
data.drop_duplicates(subset='Element 2',inplace=True)
data.reset_index(inplace=True)
data.drop(columns=['index'], inplace=True)
data.rename(columns={'Element 2':'Element'}, inplace=True)
data.to_csv(output)
