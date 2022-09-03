import re, os, sys
import numpy as np
import pandas as pd
from pprint import pprint
import gensim
import gensim.corpora as corpora
from nltk.corpus import stopwords
from joblib import dump, load


dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis, NLTKPreprocessor

corpora_dict_name = 'Corpora_Dictionary.pickle'
corpus_text_file = 'Corpus_text.pickle'
file_train = os.path.join(dirname,'complete_dataset.csv')
stop_words = stopwords.words('english')
# stop_words.extend(['shall', 'company', 'new', 'saudi', 'contract', 'contractor', 'must', 'good', 'subcontractor', 'may','schlumberger'])


corpus = get_dataframe(file_train)
print("Converting data to list ..........")
corpus['Paragraph'] = corpus['Paragraph'].astype(str)
# corpus = corpus[0:100]
data = corpus['Paragraph'].tolist()
print("NLTK Preprocessing list data..........")
data = NLTKPreprocessor().fit_transform(data)

def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))


data_words = list(sent_to_words(data))

print("Building bigrams .........")
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

data_words_bigrams = make_bigrams(data_words)
id2word = corpora.Dictionary(data_words_bigrams)
id2word.save(os.path.join(dirname,corpora_dict_name))

dump(data_words_bigrams, os.path.join(dirname, corpus_text_file))


