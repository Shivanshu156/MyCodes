import os
import sys
import gensim
import numpy as np
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis
from functions import NLTKPreprocessor, Best_estimator_grid
num_topics = sys.argv[1]



ldamallet_modelname = 'lda_mallet_' + str(num_topics) + '.pickle'
fig_name = 'confusion_matrix_ldamallet_' + str(num_topics) + '_topics.png'
supervised_model = 'Supervised_svm_ldamallet_'  + str(num_topics) + 'topics.joblib'
corpora_dict_name = 'Corpora_Dictionary.pickle'
file_test = os.path.join(dirname,'labelled_dataset.csv')



df = get_dataframe(file_test)
# df = df.iloc[0:120]

ldamodel = load(os.path.join(dirname, ldamallet_modelname))
# ldamodel.print_topics(num_topics=num_topics, num_words=10)
print("LDA Model Loaded Successfully !!!")


test_corpus = df['Text'].astype(str)
test_corpus = test_corpus.tolist()
print("NLTK Preprocessing list data..........")
test_corpus = NLTKPreprocessor().fit_transform(test_corpus)


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


test_corpus = list(sent_to_words(test_corpus))

print("Building bigrams .........")
bigram = gensim.models.Phrases(test_corpus, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[test_corpus], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

test_corpus = make_bigrams(test_corpus)

id2word = load(os.path.join(dirname,corpora_dict_name))
test_corpus = [id2word.doc2bow(text) for text in test_corpus]

print("Finding features from LDA mallet model...............")
features = ldamodel[test_corpus]
new_features = []

for feature in features:
    new_feature = [item[1] for item in feature]
    new_features.append(new_feature)


labels = LabelEncoder()
labels.fit(df['Clause'].unique().tolist())
print(labels.classes_)


X = np.asarray(new_features)
Y = np.asarray(labels.transform(df['Clause']))
print(np.shape(X), np.shape(Y))
X, Y = shuffle(X, Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)

clf = SVC()
params = {'C': [1, 10, 100, 800, 1000, 1500],
           'kernel' : ['linear', 'poly', 'rbf'],
           'gamma' : ['auto']     }

print("Finding optimal SVM model ............")

optimal_model = Best_estimator_grid(X_train, y_train, clf, params, 5)
# optimal_model = load(supervised_model)
print("Optimal Model of SVM is" , optimal_model)
best_model = optimal_model.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
inv_map = {int(k): v for k, v in enumerate(labels.classes_)}
figure = cm_analysis(y_test, y_pred, labels = list(range(len(inv_map))), ymap = inv_map, figsize=(15,15))
figure.tight_layout()
figure.savefig(fig_name)
dump(optimal_model, supervised_model)
