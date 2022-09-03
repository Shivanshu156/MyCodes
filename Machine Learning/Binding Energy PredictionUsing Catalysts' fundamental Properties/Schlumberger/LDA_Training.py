import os
import sys
import gensim
from zipfile import ZipFile
import numpy as np
import pandas as pd
from joblib import dump, load
from pprint import pprint
from sklearn.svm import SVC
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


num_topics = 100
max_f = int(sys.argv[1])
pipeline_name = 'preprocess_ppl_' + str(max_f) + '.pickle'
lda_model_name = 'lda_model_' + str(max_f) + '_features.pickle'
dirname = os.path.dirname(__file__)
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis, NLTKPreprocessor

file_train = os.path.join(dirname, 'complete_dataset.csv')

#file_train_zip = os.path.join(dirname, 'complete_data_set.zip')
#with ZipFile(file_train_zip, 'r') as f:
#    x = f.open('complete_dataset.csv')
#    print(f.open('complete_dataset.csv'))
#    corpus = get_dataframe(f.open('complete_dataset.csv'))


corpus = get_dataframe(file_train)
ppl = load(os.path.join(dirname,pipeline_name))
X = ppl.transform(corpus['Paragraph'].astype(str))
print("Data vectorized by Pipeline !!!")

id_map = dict((v, k) for k, v in ppl.named_steps.vect.vocabulary_.items())
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
print("Training LDA model ................")
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=id_map, passes=40, random_state=34)
print("LDA model has been trained Successfully !!!")

dump(ldamodel, os.path.join(dirname, lda_model_name))
print("Model has been dumped Successfully !!!")