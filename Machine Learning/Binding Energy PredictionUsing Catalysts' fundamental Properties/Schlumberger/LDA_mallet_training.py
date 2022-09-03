import re, os, sys
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
import spacy
import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from joblib import dump, load

# os.environ['MALLET_HOME'] = r'C:\Users\sverma9\Desktop\Shivanshu\mallet-2.0.8'
os.environ['MALLET_HOME'] = "/home/sverma9/Contract-Analysis/mallet2/Mallet"


# num_topics = 100
num_topics = int(sys.argv[1])
dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis, NLTKPreprocessor



corpora_dict_name = 'Corpora_Dictionary.pickle'
corpus_text_file = 'Corpus_text.pickle'
ldamallet_modelname = 'lda_mallet_' + str(num_topics) + '.pickle'


id2word = load(os.path.join(dirname,corpora_dict_name))
texts = load(os.path.join(dirname,corpus_text_file))
print("Building Corpus .................")
corpus = [id2word.doc2bow(text) for text in texts]
# print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

print("Training LDAMallet model ..........")
# mallet_path = r'C:\Users\sverma9\Desktop\Shivanshu\mallet-2.0.8\bin\mallet'
mallet_path = "/home/sverma9/Contract-Analysis/mallet2/Mallet/bin/mallet"
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
# pprint(ldamallet.show_topics(formatted=False))
print("Finding Coherence score ......")
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=texts, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print("Coherence score is ", coherence_ldamallet)

dump(ldamallet, os.path.join(dirname, ldamallet_modelname))
print("Model has been dumped Successfully !!!")

