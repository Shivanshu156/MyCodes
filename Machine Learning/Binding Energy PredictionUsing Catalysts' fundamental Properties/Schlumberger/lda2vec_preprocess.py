import os, sys
import lda2vec
import numpy as np
from joblib import dump, load
from lda2vec import preprocess
from lda2vec.corpus import  Corpus

# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
dirname = os.path.dirname(__file__)
sys.path.insert(0, dirname)

from functions import get_dataframe, cm_analysis, NLTKPreprocessor


file_train = os.path.join(dirname, 'complete_dataset.csv')
corpus = get_dataframe(file_train)
# corpus = corpus.iloc[0:200]
# print("NLTK Preprocessing text ...................\n")
# texts = NLTKPreprocessor().fit_transform(corpus['Paragraph'].astype(str))
# dump(texts,os.path.join(dirname, 'NLTK_Preprocessed_text_list.txt') )
# print("NLTK_Preprocessed Data list has been dumped successfully.............. ")
texts = load(os.path.join(dirname, 'NLTK_Preprocessed_text_list.txt'))
print("NLTK Preprocessed text list loaded .................")
for text in texts:
    if text=='':
        texts.remove(text)

max_length = 10000   # Limit of 10k words per document
tokens, vocab = preprocess.tokenize(texts, max_length, merge=False,  n_threads=4)
corpus = Corpus()
corpus.update_word_count(tokens)
corpus.finalize()

# The tokenization uses spaCy indices, and so may have gaps
# between indices for words that aren't present in our dataset.
# This builds a new compact index
compact = corpus.to_compact(tokens)

# Remove extremely rare words
pruned = corpus.filter_count(compact, min_count=10)

# Convert the compactified arrays into bag of words arrays
bow = corpus.compact_to_bow(pruned)

# Words tend to have power law frequency, so selectively
# downsample the most prevalent words
clean = corpus.subsample_frequent(pruned)

# Now flatten a 2D array of document per row and word position
# per column to a 1D array of words. This will also remove skips
# and OoV words
doc_ids = np.arange(pruned.shape[0])
flattened, (doc_ids,) = corpus.compact_to_flat(pruned, doc_ids)
assert flattened.min() >= 0

# Fill in the pretrained word vectors
n_dim = 300
# fn_wordvc = '/home/sverma9/Contract-Analysis/lda2vec/GoogleNews-vectors-negative300.bin'
fn_wordvc = os.path.join(dirname,'GoogleNews-vectors-negative300.bin')
print("Calculating Vectors .....................\n")
vectors, s, f = corpus.compact_word_vectors(vocab, filename=fn_wordvc, top=100)
# Save all of the preprocessed files
print("Dumping all processed data ...................\n")
dump(vocab,os.path.join(dirname, 'vocab.pkl'))
dump(corpus, os.path.join(dirname, 'corpus.pkl'))
np.save(os.path.join(dirname,"flattened"), flattened)
np.save(os.path.join(dirname,"doc_ids"), doc_ids)
np.save(os.path.join(dirname,"pruned"), pruned)
np.save(os.path.join(dirname,"bow"), bow)
np.save(os.path.join(dirname,"vectors"), vectors)
