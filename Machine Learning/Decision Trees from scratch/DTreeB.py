from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import sys

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
X_val = val_data[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15",
                      "X16", "X17", "X18", "X19", "X20", "X21", "X22", "X23"]]
Y_val = val_data[['Y']].values
trained_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, min_samples_split=100, min_samples_leaf=10)
trained_classifier = trained_classifier.fit(X_train, Y_train)
Y_val_predicted = trained_classifier.predict(X_val)
# print(Y_val_predicted.shape)
# print(Y_val.shape)


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
print('Accuracy for validation set is {} %'.format(accuracy*100))
