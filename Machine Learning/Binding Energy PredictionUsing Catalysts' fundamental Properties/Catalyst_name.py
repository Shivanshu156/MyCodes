import pandas as pd
import os

filename = 'catalyst_data (2).csv'
output_file = '111_C_ratio3.csv'

data = pd.read_csv(filename)

data['chemicalComposition'] = data['chemicalComposition'].str.split('3')
data['chemicalComposition'] = data['chemicalComposition'].str.join('')
data['chemicalComposition'] = data['chemicalComposition'].str.replace("9","3")

data.to_csv(output_file)