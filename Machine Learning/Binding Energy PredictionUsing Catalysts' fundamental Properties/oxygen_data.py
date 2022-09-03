import pandas as pd
import os, re

filename = 'features.csv'
file2 = 'O_111_raw_data.csv'
output = 'O_111_featured_data.csv'
df = pd.read_csv(file2)
l = df.values.tolist()

for i in range(len(l)):
    m = re.findall(r"[^\W\d_]+|\d+", l[i][0])
    m.insert(0,l[i][1])
    l[i]=m

for i in range(len(l)):
    l[i][0].insert(0,l[i][1])
    del l[i][1]
    l[i] = l[i][0]
    
b = l

for i in range(len(b)):
    if len(b[i])>3:
        if b[i][2]<b[i][4]:
            t = b[i][1]
            b[i][1] = b[i][3]
            b[i][3] = t
            t = b[i][4]
            b[i][4] = b[i][2]
            b[i][2] = t

for i in range(len(b)):
    if len(b[i])>3:
        b[i][2] = int(b[i][2])/int(b[i][4])
        del b[i][4]

df1 = pd.DataFrame(l,columns=['B.E.','A','a', 'B', 'b'] )
df1.sort_values(by=['a'], ascending=[1])

df2 = pd.DataFrame(b,columns=['B.E.','A','Ratio', 'B'])

df2.dropna()
features = pd.read_csv(filename)
features = features.drop(columns=['Unnamed: 0'])

features = features.drop(columns=['14','15', '16','17'])

list1 = features.values.tolist()
list2 = df2.values.tolist()

d = {}

for row in list1:
    key = row[0]
    del row[0]
    d[key] = row

for i in range(len(list2)):
    if list2[i][1] in d.keys():
        for value in d[list2[i][1]]:
            list2[i].append(value)
    if list2[i][3] in d.keys():
        for value in d[list2[i][3]]:
            list2[i].append(value)

col =['B.E.', 'Element1', 'Ratio', 'Element2',  'AN',
       'AM', 'G', 'P', 'R', 'EN', 'M.P.', 'B.P.', 'H_FUS', 'DENSITY', 'IE',
       'SE', "AN'", "AM'", "G'", "P'", "R'", "EN'", "M.P.'", "B.P.'", "H_FUS'",
       "DENSITY'", "IE'", "SE'"]

df4 = pd.DataFrame(list2, columns=col)
df4.dropna(inplace=True)

df4.to_csv(output)
