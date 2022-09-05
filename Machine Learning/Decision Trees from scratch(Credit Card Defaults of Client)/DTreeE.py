import pandas as pd
import numpy as np
import sys
from sklearn.tree import DecisionTreeClassifier

# train_file = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/credit-cards.train.csv'
train_file = sys.argv[1]
# test_file = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/credit-cards.test.csv'
test_file = sys.argv[2]
# val_file = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/credit-cards.val.csv'
val_file = sys.argv[3]
train_data = pd.read_csv(train_file)
train_data = train_data.drop([0])
test_data = pd.read_csv(test_file)
test_data = test_data.drop([0])
val_data = pd.read_csv(val_file)
val_data = val_data.drop([0])

X_train = train_data[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15",
                      "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]]
Y_train = train_data["Y"]
# print(X_train.tail())

new = np.array(X_train)
new = np.append(new, np.array([['0', '0', '0', '0', '0', '8', '8', '8', '8', '8', '8', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0'],
                               ['0', '0', '0', '0', '0', '8', '8', '1', '1', '8', '8', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0'],
                               ['0', '0', '0', '0', '0', '8', '8', '8', '8', '8', '8', '0', '0', '0', '0', '0', '0',
                                '0', '0', '0', '0', '0', '0']]), axis=0)

X_train_new = pd.DataFrame({'X1': new[:, 0], 'X2': new[:, 1], 'X3': new[:, 2], 'X4': new[:, 3], 'X5': new[:, 4],
                        'X6': new[:, 5], 'X7': new[:, 6], 'X8': new[:, 7], 'X9': new[:, 8], 'X10': new[:, 9],
                        'X11': new[:, 10], 'X12': new[:, 11], 'X13': new[:, 12], 'X14': new[:, 13], 'X15': new[:, 14],
                        'X16': new[:, 15], 'X17': new[:, 16], 'X18': new[:, 17], 'X19': new[:, 18], 'X20': new[:, 19],
                        'X21': new[:, 20], 'X22': new[:, 21], 'X23': new[:, 22]})


X_train_new = pd.get_dummies(X_train_new, columns=['X3', 'X4', 'X6', 'X7', 'X8', 'X9', "X10", 'X11'])
# print(list(X_train_new))
new_Y = np.array(Y_train).reshape(-1,1)
new_Y = np.append(new_Y, [['0'], ['0'], ['0']], axis=0)
# print(np.shape(new_Y))
Y_train_new = pd.DataFrame({'Y': new_Y[:, 0]})


X_val = val_data[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15",
                      "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]]
Y_val = val_data[['Y']].values
X_val = pd.get_dummies(X_val, columns=['X3', 'X4', 'X6', 'X7', 'X8', 'X9', "X10", 'X11'])
# print(list(X_val))
trained_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=3, min_samples_split=50, min_samples_leaf=10)
trained_classifier = trained_classifier.fit(X_train_new, Y_train_new)
Y_val_predicted = trained_classifier.predict(X_val)

def accuracy(Y_val, Y_pred):
    num_correct_instances = 0
    num_incorrect_instances = 0
    for i in range(len(Y_pred)):
        if Y_val[i] == Y_pred[i]:
            num_correct_instances += 1
        else:
            num_incorrect_instances += 1

    acc = num_correct_instances/len(Y_val)
    return acc


accuracy = accuracy(Y_val, Y_val_predicted)
print('Accuracy for validation set after one hot encoding is {} %'.format(accuracy*100))
