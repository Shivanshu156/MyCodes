import json
import re
import sys
import operator
from time import time
from math import log
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, accuracy_score
from pprint import pprint
from utils import getStemmedDocuments

training_data_path = sys.argv[1]
test_data_path = sys.argv[2]
part_num = sys.argv[3]
choice = int(input("Enter 1 for stemmed and stopwords removed vocabulary\nEnter 2 for only stopwords removed vocabulary"))
en_stop = set(stopwords.words('english'))

if choice == 1:
    vocab_filename = 'vocabulary_stemmed.json'
    prob_filename = 'probability_data_stemmed.json'
elif choice == 2:
    vocab_filename = 'vocabulary_stopwords.json'
    prob_filename = 'probability_data_stopwords.json'


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
            if choice == 1:
                value = getStemmedDocuments(value)
                for word in value:
                    if word != '':
                        words_set.add(word)
                        if word in vocabulary_dict[str(key)]:
                            vocabulary_dict[str(key)][word] += 1
                        else:
                            vocabulary_dict[str(key)].update({word: 1})
                        length += 1
            if choice == 2:
                for word in value.split():
                    if word != '' and word not in en_stop:
                        words_set.add(word)
                        if word in vocabulary_dict[str(key)]:
                            vocabulary_dict[str(key)][word] += 1
                        else:
                            vocabulary_dict[str(key)].update({word: 1})
                        length += 1
            wordscount[str(key)] += length
            stopper += 1
    end = time()

    with open(vocab_filename, 'w') as outfile:
        json.dump(vocabulary_dict, outfile, sort_keys=True, indent=4)

    total = 0
    for key in range(1, 6):
        total += Prob_class[str(key)]
    for key in range(1, 6):
        Prob_class[str(key)] /= total
    dictionary['wordscount'] = wordscount
    dictionary['v'] = len(list(words_set))
    dictionary['Prob_class'] = Prob_class
    with open(prob_filename, 'w') as outfile:
        json.dump(dictionary, outfile, sort_keys=True, indent=4)
    return end-start


# print(create_vocab())
with open(prob_filename) as dt:
    P_data = json.load(dt)
P_class = P_data['Prob_class']
total_count = P_data['wordscount']
v = P_data['v']
with open(vocab_filename) as vocab:
    vocabulary = json.load(vocab)


def predict(doc):
    P_class_given_doc = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}

    for key in range(1, 6):
        value = doc
        p_word = 0
        value = re.sub('[^A-Za-z0-9 ]+', '', value).lower()
        if choice == 1:
            value = getStemmedDocuments(value)
            for word in value:
                count = vocabulary[str(key)][word] if (word in vocabulary[str(key)]) else 0
                p_word += log(count + 1) - log(total_count[str(key)] + v)
        if choice == 2:
            for word in value.split():
                if word != '' and word not in en_stop:
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


if part_num == 'd':
    start = time()
    ex, pr = test_naive_bayes(test_data_path)
    accuracy = accuracy_score(ex, pr)
    print("accuracy over given test dataset is ", accuracy)

    end = time()
    print("Time taken :", end-start)







