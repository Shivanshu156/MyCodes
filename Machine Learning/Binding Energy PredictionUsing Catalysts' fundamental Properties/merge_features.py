import pandas as pd
import os
filename = 'element_wise_features.csv'
filename2 = 'element_wise_features_2.csv'
filename3 = 'features.csv'
df1 = pd.read_csv(filename, index_col=None, header=None)
df1.drop(columns=[0], inplace=True)
df1 = df1.iloc[1:]
df1.reset_index(inplace=True)
df1.drop(columns=['index'], inplace=True)
df2 = pd.read_csv(filename2, index_col=None, header=None)
df2.drop(columns=[0], inplace=True)
df2 = df2.iloc[1:]
df2.reset_index(inplace=True)
df2.drop(columns=['index'], inplace=True)
df3 = pd.concat([df1,df2], ignore_index=True)
df3.to_csv(filename3)
