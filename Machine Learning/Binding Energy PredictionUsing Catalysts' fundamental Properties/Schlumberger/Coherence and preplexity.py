import os
import sys
import gensim
import numpy as np
import pandas as pd
from joblib import load

dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
pipeline_name = 'preprocess_ppl_10000.pickle'
lda_model_name = 'lda_model_10000_features.pickle'

sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis, NLTKPreprocessor

file_train = os.path.join(dirname, 'complete_dataset.csv')

corpus = get_dataframe(file_train)
ppl = load(os.path.join(dirname,pipeline_name))
corpus_nltk = ppl.named_steps.nltk.transform(corpus['Paragraph'].astype(str))

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(sent_to_words(corpus_nltk))
id2word = gensim.corpora.Dictionary(data_words)
corpus_new = [id2word.doc2bow(data_word) for data_word in data_words]

[[(id2word[id], freq) for id, freq in cp] for cp in corp[:10]]

ldamodel = load(os.path.join(dirname, lda_model_name))

print('perplexity is :',ldamodel.log_perplexity(corpus_new))
coherence_model_lda = gensim.models.CoherenceModel(model=ldamodel, texts=data_words, dictionary=id2word, coherence='c_v')
print("coherence score is :", coherence_model_lda.get_coherence())