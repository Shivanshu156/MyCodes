import os, sys
import pandas as pd
from itertools import groupby
from collections import Iterable

def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item


filename = 'catalyst.xls'
C_data_pure = []
C_data = pd.read_excel(filename, sheet_name='Carbon_data', index=False)
# C_data = pd.read_excel(filename, sheet_name='Oxygen data', index=False)
C_data = C_data.drop(columns=['Equation', 'activationEnergy'])
C_data = C_data.values.tolist()
for i in range(len(C_data)):
    C_data[i][0]= [''.join(g) for _, g in groupby(C_data[i][0], str.isalpha)]
    C_data[i] = list(flatten(C_data[i]))
    # C_data[i][1] = int(C_data[i][1])
    # C_data[i][3] = int(C_data[i][3])
    if int(C_data[i][3]) > int(C_data[i][1]) and len(C_data[i])==6:
        C_data[i][1] , C_data[i][3] = C_data[i][3] , C_data[i][1]
        C_data[i][0] , C_data[i][2] = C_data[i][2] , C_data[i][0]
    if len(C_data[i])==4:
        C_data_pure.append(C_data[i])

C_data = [x for x in C_data if len(x)==6]
C_data = pd.DataFrame(C_data)
C_data.columns  = ["Element 1", "a", "Element 2", "b", "facet", "Enegry"]
C_data["a"] = pd.to_numeric(C_data["a"])
C_data["b"] = pd.to_numeric(C_data["b"])
C_data['Ratio'] = C_data['a']/C_data['b']
C_data_pure = pd.DataFrame(C_data_pure)
C_data_pure.to_excel('single_atom_energy_C.xlsx', index=False, header=['Element', 'a', 'facet', 'Energy'])
# C_data_pure.to_excel('single_atom_energy_O.xlsx', index=False, header=['Element', 'a', 'facet', 'Energy'])
C_data.to_excel('splitted_atom_C_110.xlsx', index=False)
# C_data.to_excel('splitted_atom_O.xlsx', index=False)