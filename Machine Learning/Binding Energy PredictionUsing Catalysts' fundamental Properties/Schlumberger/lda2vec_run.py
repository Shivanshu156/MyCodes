import os, sys
import time
import shelve
import numpy as np
from joblib import load, dump
from lda2vec import utils
from lda2vec import prepare_topics, print_top_words_per_topic, topic_coherence

# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
dirname = os.path.dirname(__file__)
sys.path.insert(0, dirname)
from lda2vec_model import LDA2Vec

fn_vectors = os.path.join(dirname, 'vectors.npy')
fn_vocab = os.path.join(dirname, 'vocab.pkl')
fn_corpus = os.path.join(dirname, 'corpus.pkl')
fn_flatnd = os.path.join(dirname, 'flattened.npy')
fn_docids = os.path.join(dirname, 'doc_ids.npy')
fn_vectors = os.path.join(dirname, 'vectors.npy')

vocab = load(fn_vocab)
corpus = load(fn_corpus)
flattened = np.load(fn_flatnd)
doc_ids = np.load(fn_docids)
vectors = np.load(fn_vectors)

n_docs = doc_ids.max() + 1              # Number of documents
n_vocab = flattened.max() + 1           # Number of unique words in the vocabulary
clambda = 200.0                         # 'Strength' of the dircihlet prior; 200.0 seems to work well

n_topics = int(sys.argv[1])             # Number of topics to fit
power = float(os.getenv('power', 0.75)) # Power for neg sampling
temperature = float(os.getenv('temperature', 1.0))  # Sampling temperature
n_units = int(os.getenv('n_units', 300))            # Number of dimensions in a single word vector
words = corpus.word_list(vocab)[:n_vocab]           # Get the string representation for every compact key

doc_idx, lengths = np.unique(doc_ids, return_counts=True)   # How many tokens are in each document
doc_lengths = np.zeros(doc_ids.max() + 1, dtype='int32')
doc_lengths[doc_idx] = lengths

tok_idx, freq = np.unique(flattened, return_counts=True)    # Count all token frequencies
term_frequency = np.zeros(n_vocab, dtype='int32')
term_frequency[tok_idx] = freq

lda2vec_data_file = 'data_lda2vec_' + str(n_topics) + 'topics'
print("Training LDA model ................")
model = LDA2Vec(n_documents=n_docs, n_document_topics=n_topics,
                n_units=n_units, n_vocab=n_vocab, counts=term_frequency,
                n_samples=15, power=power, temperature=temperature)
                
model.sampler.W.data[:, :] = vectors[:n_vocab, :]
data = prepare_topics(model.mixture.weights.W.data,model.mixture.factors.W.data, model.sampler.W.data, words)
top_words = print_top_words_per_topic(data)
dump(data, os.path.join(dirname, lda2vec_data_file))
print("Model has been dumped Successfully !!!")