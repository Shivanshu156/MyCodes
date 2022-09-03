import os, sys
import pandas as pd

C_data = pd.read_excel('calculated_energy_O.xlsx', index = False)
C_data['Difference'] = abs(C_data['Calculated_energy'] - C_data['Enegry'])
C_data.sort_values(by = ['Element 1', 'Element 2', 'Difference'], ignore_index=True, inplace=True)
C_data.drop_duplicates(['Element 1', 'Element 2'], keep='first', ignore_index = True, inplace=True)
C_data.to_excel('Calculated_energy_nodup_O.xlsx', index=False)