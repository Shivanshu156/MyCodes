import numpy as np


def PreProcess(train_data):
    data = []
    col_names = {'X1', 'X5', 'X12', 'X13', 'X14', 'X15', 'X16', 'X17', 'X18', 'X19', 'X20', 'X21', 'X22', 'X23'}
    col_dic = {'X1': [0], 'X5': [0], 'X12': [0], 'X13': [0], 'X14': [0], 'X15': [0], 'X16': [0], 'X17': [0], 'X18': [0], 'X19': [0],
               'X20': [0], 'X21': [0], 'X22': [0], 'X23': [0]}

    for col_name in col_names:
        values = list(map(int, train_data[col_name].values))
        med = np.median(values)
        for value in values:
            if value > med:
                col_dic[col_name].append(1)
            else:
                col_dic[col_name].append(0)
    train_data.update(col_dic)
    # print(train_data.head())
    for index, row in train_data.iterrows():
        data.append(row)

    return data
