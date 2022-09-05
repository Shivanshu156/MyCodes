import sys
import random
from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.utils.multiclass import unique_labels
from matplotlib import pyplot as plt
from pprint import pprint
from sklearn.utils import shuffle

# Taking input parameters

config_file = sys.argv[1]
# config_file = 'config.txt'
one_hot_train = sys.argv[2]
# one_hot_train = '/Users/shivanshu/Desktop/one_hot_train.csv'
one_hot_test = sys.argv[3]
# one_hot_test = '/Users/shivanshu/Desktop/one_hot_test.csv'
# Reading configure file parameters
epoch_no = int(input('Enter the number of epochs for training\t'))
file = open(config_file, 'r')
n_input = int(file.readline().split()[0])
n_output = int(file.readline().split()[0])
batch_size = int(file.readline().split()[0])  # batch size
n_hidden = int(file.readline().split()[0])
h = []
for word in file.readline().split():
    h.append(int(word))

# print('perceptron units are ', h)
non_linearity = file.readline().split()[0]
lr = file.readline().split()[0]
size = [n_input]
for neurons in h:
    size.append(neurons)
size.append(n_output)
print('Total layers are ', size)
# processing data
train_data = pd.read_csv(one_hot_train)
test_data = pd.read_csv(one_hot_test)
# train_data = train_data[:1000]
# test_data = test_data[:1000]
x_col = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
y_col = ['10_0', '10_1', '10_2', '10_3', '10_4', '10_5', '10_6', '10_7', '10_8', '10_9']


class Network(object):
    def __init__(self, sizes):
        # sizes is the list of number of neurons in layers of the neural network
        self.num_of_layers = len(sizes)
        self.layers = sizes
        self.weights = [np.ones((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]
        # list of weights or theta matrix for each layer
        # print('Weights are ',self.weights)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # list of (-threshold) for each layer except the input layer

    def mbgd(self, training_data, epoch, eta, mini_batch_size, test_data=None):
        n = len(training_data)

        for i in range(epoch):
            # random.shuffle(training_data)
            start = time()
            training_data = shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update(mini_batch, eta)
            end = time()
            print("Epoch {} is completed in time {}".format(i+1,end - start))
            if test_data is not None:

                r1, r= self.evaluate(test_data)
                acc = accuracy_score(r, r1)
                print('Accuracy is', acc)

    def forward_prop(self, x):
        # to return output of network for input layer x
        y = x
        for w, b in zip(self.weights, self.biases):
            y = sigmoid(np.dot(w, y) + b)
        return y

    def update(self, mini_batch, eta):
        n = len(mini_batch)
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        for index, row in mini_batch.iterrows():
            x = np.asarray(row[x_col]).reshape(-1,1)
            # print("X is ", x.T)
            y = np.asarray(row[y_col]).reshape(-1,1)
            # print("Y is ", y.T)
            delta_w, delta_b = self.back_prop(x, y)
            grad_w = [grad + delta for grad, delta in zip(grad_w, delta_w)]
            grad_b = [grad + delta for grad, delta in zip(grad_b, delta_b)]
        # print('Updated gradient w is ',grad_w)
        self.weights = [w - (eta / n) * grad for w, grad in zip(self.weights, grad_w)]
        self.biases = [b - (eta / n) * grad for b, grad in zip(self.biases, grad_b)]

    def back_prop(self, x, y):
        # print(np.shape(x), 'x shape')
        # print(np.shape(y), 'y shape')
        grad_w = [np.zeros(w.shape) for w in self.weights]
        grad_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zl = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activation) + b
            zl.append(z)
            # print(np.shape(z), 'z shape')
            activation = sigmoid(z)
            # print(np.shape(activation), ' activation shape')
            activations.append(activation)
        # print(type(activation[-1]),type(y))
        # print(np.shape(activation[-1]), len(y))
        delta = (activations[-1] - y) * sigmoid_prime(zl[-1])
        grad_w[-1] = np.dot(delta, activations[-2].transpose())
        grad_b[-1] = delta
        for i in range(2, self.num_of_layers):
            z = zl[-i]
            delta = np.dot(self.weights[-i + 1].transpose(), delta) * sigmoid_prime(z)
            grad_w[-i] = np.dot(delta, activations[-i - 1].transpose())
            grad_b[-i] = delta
        return grad_w, grad_b

    def evaluate(self, test_data):
        result = []
        result2 = []
        sum = 0
        for index, row in test_data.iterrows():
            x = np.asarray(row[x_col].values).reshape(-1,1)
            y = np.array(row[y_col]).reshape(-1,1)
            y1 = self.forward_prop(x)

            result.append(y.argmax())
            result2.append(y1.argmax())
            # if np.array_equal(y2, y):
            #     sum += 1

        return result2, result


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def plot_confusion_matrix(y_true, y_pred, normalize=False, title=None, cmap=plt.cm.Blues):
    
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    print('Confusion matrix is >>>>>>>>>>>>>>>')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if non_linearity == 'sigmoid' and lr == 'fixed':
    neural_network = Network(size)
    start = time()
    neural_network.mbgd(train_data, epoch_no, 0.1, batch_size)
    end = time()
    print("Neural Network Created Successfully !!!\nTime Taken is {}".format(end-start))

    # r1, r = neural_network.evaluate(train_data)
    # acc = accuracy_score(r, r1)
    # print('Training Accuracy ', acc)
    print("Testing on given test dataset >>>>>>>>>>")
    r12, r2 = neural_network.evaluate(test_data)
    acc = accuracy_score(r2, r12)
    print('Test accuracy is ', acc)

    plot_confusion_matrix(r2, r12, title='Confusion matrix')
    plt.show()

if non_linearity == 'sigmoid' and lr == 'variable':
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(activation='logistic',max_iter=2000, batch_size=100, hidden_layer_sizes=(20,20), learning_rate='adaptive',
                    learning_rate_init=0.1, shuffle=True, solver='sgd', tol=0.0001)
    X = train_data[x_col]
    X = X.astype(np.float)
    Y = train_data[y_col].values
    Y = Y.astype(np.float)
    clf.fit(X, Y)
    X_test = test_data[x_col]
    X_test = X_test.astype(np.float)
    Y_test = test_data[y_col].values
    # print(Y_test.head())
    Y_test = Y_test.astype(np.float)

    Y_pred = clf.predict(X_test)
    Y_pred = Y_pred.astype(np.float)
    result = []
    result1 = []
    for i in range(len(Y_pred)):
        result1.append(np.argmax(Y_pred[i]))
    for i in range(len(Y_test)):
        result.append(np.argmax(Y_test[i]))

    acc = accuracy_score(result, result1)
    print('Test accuracy is', acc)
    plot_confusion_matrix(result, result1)
    plt.show()



