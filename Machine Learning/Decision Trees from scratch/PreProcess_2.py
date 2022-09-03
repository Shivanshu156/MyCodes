import sys
import pandas as pd

train_path = sys.argv[1]
# train_path = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/train_poker.csv'
test_path = sys.argv[2]
# test_path = '/Users/shivanshu/Desktop/Sem8/ML/Assignment3/test_pocker.csv'
one_hot_train = sys.argv[3]
# one_hot_train = '/Users/shivanshu/Desktop/one_hot_train.csv'
one_hot_test = sys.argv[4]
# one_hot_test = '/Users/shivanshu/Desktop/one_hot_test.csv'

train_data = pd.read_csv(train_path, header=None)
test_data = pd.read_csv(test_path, header=None)

train_data = pd.get_dummies(train_data, columns=[10])
test_data = pd.get_dummies(test_data, columns=[10])

train_data.to_csv(one_hot_train, index=False)
test_data.to_csv(one_hot_test, index=False)


