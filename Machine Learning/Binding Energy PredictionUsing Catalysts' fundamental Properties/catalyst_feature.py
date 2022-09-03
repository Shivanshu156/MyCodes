import pandas as pd

input1 = '111_C_ratio3.csv'
input2 = 'data_clean_added_C.csv'
output = 'featured_data_111_C.csv'

data1 = pd.read_csv(input1)
data2 = pd.read_csv(input2)
data1.dropna(inplace = True)
data1 = data1[['Equation', 'chemicalComposition', 'facet', 'reactionEnergy']]
data1.rename(columns = {'chemicalComposition' : 'Alloy'}, inplace = True)

data3 = pd.merge(data1, data2, on='Alloy')
data3.to_csv(output)

