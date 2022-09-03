import os
import sys
import gensim
from zipfile import ZipFile
import numpy as np
import pandas as pd
from joblib import dump
from pprint import pprint
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer


dirname = os.path.dirname(__file__)
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis, NLTKPreprocessor

file_train = os.path.join(dirname, 'complete_dataset.csv')

max_f = int(sys.argv[1])
corpus = get_dataframe(file_train)
pipeline_name = 'preprocess_ppl_' + str(max_f) + '.pickle'

sw1 = stopwords.words("english")
sw1.extend(['shall', 'company', 'new', 'saudi', 'contract', 'contractor', 'must', 'good', 'subcontractor', 'may', 'schlumberger'])

ngram_range = (1, 2)

ppl = Pipeline([
        ('nltk',NLTKPreprocessor()),
        ('vect', CountVectorizer(max_df=.5, analyzer='word', ngram_range=ngram_range, max_features = max_f))
        ] )

ppl.fit(corpus['Paragraph'].astype(str))
print("Pipeline Trained !!!")
dump(ppl, os.path.join(dirname, pipeline_name))

print("Preprocessing Pipeline has been dumped Successfully !!!")
sys.exit()
