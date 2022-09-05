import json
import sys
import re
import random
import operator
from time import time
from math import log
from sklearn.metrics import confusion_matrix, accuracy_score
from pprint import pprint

training_data_path = sys.argv[1]
test_data_path = sys.argv[2]
part_num = sys.argv[3]


def create_vocab():
    wordscount = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    Prob_class = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    words_set = set()
    vocabulary_dict = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}}
    dictionary = {'wordscount': {}, 'v': 0, 'Prob_class': {}}
    stopper = 0
    start = time()
    for line in open(training_data_path, 'r'):
        if stopper < 600000:
            line = json.loads(line)
            key = int(line['stars'])
            value = line['text']
            Prob_class[str(key)] += 1
            length = 0
            value = re.sub('[^A-Za-z0-9 ]+', '', value)
            words = [u for u in re.split('\s+', value)]
            for word in words:
                if word != '':
                    word = word.lower()
                    words_set.add(word.lower())
                    if word in vocabulary_dict[str(key)]:
                        vocabulary_dict[str(key)][word] += 1
                    else:
                        vocabulary_dict[str(key)].update({word: 1})
                    length += 1
            wordscount[str(key)] += length
            stopper+=1
    end = time()
    with open('vocabulary.json', 'w') as outfile:
        json.dump(vocabulary_dict, outfile, sort_keys=True, indent=4)

    total = 0
    for key in range(1, 6):
        total += Prob_class[str(key)]
    for key in range(1, 6):
        Prob_class[str(key)] /= total
    dictionary['wordscount'] = wordscount
    dictionary['v'] = len(list(words_set))
    dictionary['Prob_class'] = Prob_class
    with open('probability_data.json', 'w') as outfile:
        json.dump(dictionary, outfile, sort_keys=True, indent=4)
    # return , , Prob_class,vocabulary_dict
    return end-start


print("Assignment 2 Question 1 part ",part_num)
# print(create_vocab())
with open('probability_data.json') as dt:
    P_data = json.load(dt)
P_class = P_data['Prob_class']
total_count = P_data['wordscount']
v = P_data['v']
with open('vocabulary.json') as vocab:
    vocabulary = json.load(vocab)


def predict(doc):

    P_class_given_doc = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

    for key in range(1, 6):
        p_word = 0
        doc = re.sub('[^A-Za-z0-9 ]+', '', doc).lower()
        for word in doc.split():
            count = vocabulary[str(key)][word] if (word in vocabulary[str(key)]) else 0
            p_word += log(count + 1) - log(total_count[str(key)] + v)
        P_class_given_doc[str(key)] = p_word + log(P_class[str(key)])
    predicted_class = max(P_class_given_doc.items(), key=operator.itemgetter(1))[0]
    return float(predicted_class)


def test_naive_bayes(test_data_path):
    expected = []
    prediction = []

    for line in open(test_data_path, 'r'):
        line = json.loads(line)
        expected.append(float(line['stars']))
        prediction.append(predict(line['text']))
    return expected, prediction


def random_results(test_datapath):
    data = []
    expected = []
    prediction = []
    for line in open(test_datapath, 'r'):
        data.append(json.loads(line))

    for line in data:
        expected.append(float(line['stars']))
        prediction.append(float(random.randint(1, 5)))

    return expected, prediction


if part_num == 'a':
    start = time()
    ex, pr = test_naive_bayes(test_data_path)
    accuracy = accuracy_score(ex, pr)
    print("Accuracy for given test file is ", accuracy)
    end = time()
    print("Time taken :", end-start)

if part_num == 'b':
    start = time()
    ex, pr = random_results(test_data_path)
    accuracy = accuracy_score(ex, pr)
    print("accuracy over test data set guessing randomly is ", accuracy)
    accuracy = P_class[max(P_class.items(), key=operator.itemgetter(1))[0]]
    print("accuracy over test data set using majority prediction is ", accuracy)
    end = time()
    print("Time taken :", end-start)

if part_num == 'c':
    start = time()
    ex, pr = test_naive_bayes(test_data_path)
    results = confusion_matrix(ex, pr)
    print("Confusion matrix for test data set : ")
    pprint(results)
    end = time()
    print("Time taken :", end-start)





