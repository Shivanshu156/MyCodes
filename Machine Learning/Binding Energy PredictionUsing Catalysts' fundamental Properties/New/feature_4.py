import os, sys
import pandas as pd

input1 = 'element_wise_features.csv'
input2 = 'element_wise_features_2.csv'
filename = 'Calculated_energy_nodup_O.xlsx'

C_data = pd.read_excel(filename,index = False)
# C_data['Upper_limit']=C_data['Upper_limit']-.2
# C_data['Lower_limit']=C_data['Lower_limit']+.2
# C_data = C_data[C_data['Calculated_energy']<=C_data['Upper_limit']]
# C_data = C_data[C_data['Calculated_energy']>=C_data['Lower_limit']]

features = pd.read_csv(input2, index_col=False)
extra_features = pd.read_csv(input1, index_col=False)
features.drop('Unnamed: 0', axis=1, inplace=True)
extra_features.drop('Unnamed: 0', axis=1, inplace=True)
# features.values.tolist()
# C_data.values.tolist()
extra_features = extra_features[['Element', 'Pauling', 'WorkFunction', 'dBandCenter']]
features.rename(columns={"Element" : "Element 1"}, inplace=True)
data = pd.merge(C_data, features, on='Element 1', how='outer')

features.rename(columns={"Element 1" : "Element 2"}, inplace=True)
data = pd.merge(data, features, on='Element 2', how='outer')
data.dropna(inplace=True)
extra_features.rename(columns={"Element" : "Element 1"}, inplace=True)
data = pd.merge(data, extra_features, on='Element 1', how='outer')
extra_features.rename(columns={"Element 1" : "Element 2"}, inplace=True)
data = pd.merge(data, extra_features, on='Element 2', how='outer')
data.to_excel('Featured_data_O_111.xlsx', index=False)
# data.to_excel('Featured_data_C_111_refined_80.xlsx', index=False)