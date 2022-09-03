import sys
import math
import random
import operator
import numpy as np
import pandas as pd
from time import time
from itertools import groupby
import matplotlib.pyplot as plt
from PreProcess import PreProcess


# train_file = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/credit-cards.train.csv'
train_file = sys.argv[1]
# test_file = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/credit-cards.test.csv'
test_file = sys.argv[2]
# val_file = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/credit-cards.val.csv'
val_file = sys.argv[3]
train_data = pd.read_csv(train_file, header=0)
train_data = train_data.drop([0])
test_data = pd.read_csv(test_file, header=0)
test_data = test_data.drop([0])
val_data = pd.read_csv(val_file, header=0)
val_data = val_data.drop([0])
train_data = PreProcess(train_data)
test_data = PreProcess(test_data)
val_data = PreProcess(val_data)

columns = {'X1': [0, 1], 'X2': [1, 2], 'X3': [0, 1, 2, 3, 4, 5, 6], 'X4': [0, 1, 2, 3], 'X5': [0, 1],
           'X6': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X7': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           'X8': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X9': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           'X10': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'X11': [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
           'X12': [0, 1],'X13': [0, 1], 'X14': [0, 1], 'X15': [0, 1], 'X16': [0, 1], 'X17': [0, 1], 'X18': [0, 1],
           'X19': [0, 1], 'X20': [0, 1], 'X21': [0, 1], 'X22': [0, 1], 'X23': [0, 1]}


class TreeNode:
    def __init__(self, col, split_data, prediction, node_no):
        self.col = col
        self.split_data = split_data
        self.prediction = prediction
        self.node_no = node_no
        self.parent = None
        self.children = {-2: None, -1: None, 0: None, 1: None, 2: None, 3: None, 4: None, 5: None,
                         6: None, 7: None, 8: None, 9: None}


def getBestSplit(data):
    maxInfoGain = -float('inf')
    splited_data = {}
    best_col = ''
    checkinfogain = 0
    for col in columns:
        split_values = columns[col]
        infoGain, splited_data_dic, prediction = calcInfoGain(data, col, split_values)
        checkinfogain += infoGain
        if infoGain > maxInfoGain and infoGain != 0:
            maxInfoGain = infoGain
            splited_data = splited_data_dic
            best_col = col
    if checkinfogain == 0:
        return None, None, None
    return best_col, splited_data, prediction


def calcInfoGain(data, col, split):
    totalLen = len(data)
    infoGain = entropy(data)
    splited_data, prediction = getDataSplit(data, split, col)
    for key in splited_data:
        infoGain -= len(splited_data[key])/totalLen * entropy(splited_data[key])
    return infoGain, splited_data, prediction


def getDataSplit(data, split, col):

    splited_data = {}
    for split_value in split:
        for row in data:
            if int(row[col]) == split_value:
                if split_value in splited_data.keys():
                    splited_data[split_value].append(row)
                else:
                    splited_data.update({split_value: []})
                    splited_data[split_value].append(row)
    group_by_class = groupby(data, lambda x: x['Y'])
    grp_len = {'0': 0, '1': 0}
    for key, group in group_by_class:
        grp_len[key] += len(list(group))
    prediction = max(grp_len.items(), key=operator.itemgetter(1))[0]
    return splited_data, prediction


def entropy(data):
    total_len = len(data)
    entropy = 0
    grp_len = {0: 0, 1: 0}
    group_by_class = groupby(data, lambda x: int(x['Y']))
    for key, group in group_by_class:
        grp_len[key] += len(list(group))
    for key in grp_len:
        if grp_len[key] == 0:
            entropy = 0
        else:
            entropy += -(grp_len[key] / total_len) * math.log((grp_len[key] / total_len), 2)
    return entropy


def build_tree(data, node_no):

    if len(data) == 0:
        node = TreeNode(None, None, random.choice(['0', '1']), node_no+1)
        node_no += 1

    else:
        group_by_class = groupby(data, lambda x: x['Y'])
        grp_len = {'0': 0, '1': 0}
        for key, group in group_by_class:
            grp_len[key] += len(list(group))

    # if max(grp_len['0']/len(data), grp_len['1']/len(data)) > 0.95:
    #     if grp_len['0'] > grp_len['1']:
    #         node = TreeNode(None, None, '0', node_no+1)
    #         node_no += 1
    #     else:
    #         node = TreeNode(None, None, '1', node_no+1)
    #         node_no += 1

        if grp_len['1'] == 0:
            node = TreeNode(None, None, '0', node_no + 1)
            node_no += 1
        elif grp_len['0'] == 0:
            node = TreeNode(None, None, '1', node_no + 1)
            node_no += 1

        else:
            bestsplit_col, bestsplit_data, prediction = getBestSplit(data)

            if bestsplit_col is not None:
                node = TreeNode(bestsplit_col, bestsplit_data, prediction, node_no+1)
                node_no += 1
                for key in bestsplit_data.keys():
                    node.children[key], node_no = build_tree(bestsplit_data[key], node_no)
                    node.children[key].parent = node
            else:
                if grp_len['0'] > grp_len['1']:
                    node = TreeNode(None, None, '0', node_no+1)
                    node_no += 1
                else:
                    node = TreeNode(None, None, '1', node_no+1)
                    node_no += 1
    # if total_nodes < node_no:
    #     total_nodes = node_no

    return node, node_no


def classify(tree, row, node_limit):
    if tree.node_no <= node_limit:
        if tree.col is None:
            num = int(tree.prediction)
        else:
            if tree.children[int(row[tree.col])] is not None:
                tree = tree.children[int(row[tree.col])]
                if tree.node_no <= node_limit:
                    num = classify(tree, row, node_limit)
                else:
                    num = tree.parent.prediction
            else:
                num = tree.prediction
    else:
        num = tree.parent.prediction
    return num


def test_tree(tree, data, total_nodes):
    total_len = len(data)
    node_list = []
    accuracy_list = []
    nvm = 1
    while nvm < total_nodes+1:
        num_correct_instances = 0
        num_incorrect_instances = 0
        for row in data:
            actual_class = row['Y']
            predicted_class = classify(tree, row, nvm)
            if int(actual_class) == int(predicted_class):
                num_correct_instances = num_correct_instances + 1
            else:
                num_incorrect_instances = num_incorrect_instances + 1
        node_list.append(nvm)
        accuracy_list.append((num_correct_instances / total_len) * 100)
        nvm += 100
    return node_list, accuracy_list


print("Please wait while creating Decision Tree")
start = time()
model, no_of_nodes = build_tree(train_data, 0)
end = time()
print("Tree built successfully in", end - start, "sec")
print("Total no of nodes are ", no_of_nodes)
print("Testing on validation set....")
start = time()
X2, Y2 = test_tree(model, val_data, no_of_nodes)
end = time()
print("Time taken : ", end-start)
print("Testing on test set....")
start = time()
X1, Y1 = test_tree(model, test_data, no_of_nodes)
end = time()
print("Time taken : ", end-start)
print("Testing on training set....")
start = time()
X, Y = test_tree(model, train_data, no_of_nodes)
end = time()
print("Time taken : ", end-start)

X = np.array(X)
Y = np.array(Y)
X1 = np.array(X1)
Y1 = np.array(Y1)
X2 = np.array(X2)
Y2 = np.array(Y2)
X3 = [X, Y, X1, Y1, X2, Y2]
X3 = np.array(X3).T
# print(np.shape(X3))
with open('A3Q1a.csv', 'wb') as file:
    np.savetxt(file, X3, delimiter=',')
plt.plot(X, Y, 'r')
plt.plot(X1, Y1, 'yo')
plt.plot(X2, Y2, 'bo')
plt.ylabel("Accuracy in %", fontsize=15)
plt.xlabel("No of nodes ", fontsize=15)
plt.title("A3Q1(a)", fontsize=15)
plt.ylim(65, 95)
plt.show()
#
# print("total no of nodes is ", no_of_nodes)
