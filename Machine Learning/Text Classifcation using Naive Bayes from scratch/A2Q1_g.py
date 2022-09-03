import json
import re
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import bigrams
import operator
from time import time
from math import log
from sklearn.metrics import accuracy_score, f1_score


training_data_path = sys.argv[1]
test_data_path = sys.argv[2]
part_num = sys.argv[3]
en_stop = set(stopwords.words('english'))
vocabfilename = 'vocabulary_full_train.json'
prob_filename = 'probability_data_full_train.json'


def create_vocab():
    featurescount = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    Prob_class = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    features_set = set()
    vocabulary_dict = {"1": {}, "2": {}, "3": {}, "4": {}, "5": {}}
    dictionary = {'featurescount': {}, 'v': 0, 'Prob_class': {}}
    stopper = 0
    start = time()
    for line in open(training_data_path, 'r'):

        line = json.loads(line)
        key = int(line['stars'])
        value = line['text']
        Prob_class[str(key)] += 1
        length = 0
        value = re.sub('[^A-Za-z0-9 ]+', '', value)
        words = word_tokenize(value.lower())
        words = filter(lambda word: word not in en_stop, words)
        features = bigrams(words)
        for feature in features:
            feature = '_'.join(feature)
            features_set.add(feature)
            if feature in vocabulary_dict[str(key)]:
                vocabulary_dict[str(key)][feature] += 1
            else:
                vocabulary_dict[str(key)].update({feature: 1})
            length += 1
        featurescount[str(key)] += length
    end = time()
    with open(vocabfilename, 'w') as outfile:
        json.dump(vocabulary_dict, outfile, sort_keys=True, indent=4)

    total = 0
    for key in range(1, 6):
        total += Prob_class[str(key)]
    for key in range(1, 6):
        Prob_class[str(key)] /= total
    dictionary['featurescount'] = featurescount
    dictionary['v'] = len(list(features_set))
    dictionary['Prob_class'] = Prob_class
    with open(prob_filename, 'w') as outfile:
        json.dump(dictionary, outfile, sort_keys=True, indent=4)
    return end-start


# print(create_vocab())
print("Assignment 2 Question 1 part ",part_num)
with open(prob_filename) as dt:
    P_data = json.load(dt)
P_class = P_data['Prob_class']
total_count = P_data['featurescount']
v = P_data['v']
with open(vocabfilename) as vocab:
    vocabulary = json.load(vocab)


def predict(doc):

    P_class_given_doc = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

    for key in range(1, 6):
        p_word = 0
        doc = re.sub('[^A-Za-z0-9 ]+', '', doc).lower()
        words = word_tokenize(doc)
        words = filter(lambda word: word not in en_stop, words)
        features = bigrams(words)
        for feature in features:
            feature = '_'.join(feature)
            count = vocabulary[str(key)][feature] if (feature in vocabulary[str(key)]) else 0
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


if part_num == 'g':
    start = time()
    ex, pr = test_naive_bayes(test_data_path)
    accuracy = accuracy_score(ex, pr)
    print("Accuracy over test dataset with full trained model using bigram is ", accuracy)
    f_mat = f1_score(ex, pr, average='macro')
    print("F1 score on Macro avg is ", f_mat)
    end = time()
    print("Time taken :", end - start)
