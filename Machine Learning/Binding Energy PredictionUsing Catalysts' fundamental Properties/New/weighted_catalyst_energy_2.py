import os, sys
import pandas as pd

C_data = pd.read_excel('splitted_atom_C.xlsx')
# C_data = pd.read_excel('splitted_atom_O.xlsx')
C_data_pure = pd.read_excel('single_atom_energy.xlsx')
# C_data_pure = pd.read_excel('single_atom_energy_O.xlsx')
C_data = C_data[C_data['facet']==111]
C_data.reset_index(inplace=True)
C_data['Calculated_energy'] = 0

for i in range(len(C_data)) : 
    x,y = 0,0
    for j in range(len(C_data_pure)):
        if C_data_pure.loc[j, "Element"] == C_data.loc[i,"Element 1"]:
            x = C_data_pure.loc[j, "Energy"]
        if C_data_pure.loc[j, "Element"] == C_data.loc[i,"Element 2"]:
            y = C_data_pure.loc[j, "Energy"]
    if x!=0 and y!=0:
        C_data.loc[i, "Calculated_energy"] = (C_data.loc[i, "a"]*x + C_data.loc[i, "b"]*y)/12

C_data = C_data[C_data['Calculated_energy']!=0]
C_data.reset_index(inplace=True)
C_data.drop(['level_0','index'],axis=1, inplace=True)
C_data.to_excel('calculated_energy_O.xlsx', index=False)
C_data.sort_values(by = ['Element 1', 'Element 2'], ignore_index=True)
dup = C_data.duplicated(['Element 1', 'Element 2'], keep=False)
    