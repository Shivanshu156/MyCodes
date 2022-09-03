import os
import re
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from joblib import load


# embeddings_folder  = r"C:\Users\ABansal4\Desktop\contract_review_poc\Final Codes\embeddings"
# #embeddings_folder = r"/home/KJagtap/contract-review-tool-server/python_script/embeddings"
# project_folder_1 = os.path.dirname(embeddings_folder)


# ##Embedding paths
# ## GLOVE MODEL
# glove_folder  = os.path.join(embeddings_folder, 'glove.6B')
# glove_file = 'glove.6B.300d.pickle'
# glove_data_file = os.path.join(glove_folder, glove_file)

# ##Fasttext
# fasttext_folder  = os.path.join(embeddings_folder, 'fasttext')
# fasttext_file = 'wiki.simple.vec'
# fasttext_data_file = os.path.join(fasttext_folder, fasttext_file)

# ##conceptNet
# conceptNet_folder  = os.path.join(embeddings_folder, 'conceptNet')
# conceptNet_file = 'numberbatch-en.pickle'
# conceptNet_data_file = os.path.join(conceptNet_folder, conceptNet_file)

## cleaning Dictionary
re_repl = {
#     r"\bdon't\b": "do not",
#     r"\bdoesn't\b": "does not",
#     r"\bdidn't\b": "did not",
#     r"\bhasn't\b": "has not",
#     r"\bhaven't\b": "have not",
#     r"\bhadn't\b": "had not",
#     r"\bwon't\b": "will not",
#     r"\bwouldn't\b": "would not",
#     r"\bcan't\b": "can not",
#     r"\bcannot\b": "can not",
#     r"\b\"\"\"\b": " ",
#     r"\b\&quot;\b": " ",
#     r"\b\/\b": " ",r'\d+' : "",
#     r'[^\w\s]' : "",
#     r'\b[a-zA-Z]\b' : "",
    r'($chlumberger|schiumberger|schiurnmerger|sohlumberger|schlulfibÃ¿rger|schlumbder|\
    schlumbejsger|schlumjaerger|schlunibergei|schlurplerger|schlumheruer|schlumhergel)' : 'company',
    'date' : "",
    'http' : "",
    'https' : "",
    'www' : "",
    'schlumberger' : 'company',
    'Schlumberger' : 'company',
    'slb' : 'company',
    '(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten| \
          eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen| \
          eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty| \
          ninety|hundred|thousand)' : ""
}
#re_repl = load(os.path.join(project_folder_1, 'spelling_dict.pickle'))


## Pos Tag Count
from textstat import textstat
from textblob import TextBlob
from nltk import sent_tokenize, pos_tag, wordpunct_tokenize

class LinguisticVectorizer(BaseEstimator, TransformerMixin):
    Cnt = 0
    
    def get_feature_names(self):
        return np.array(['nouns', 'adjectives', 'verbs', 'adverbs',
                         'allcaps', 'exclamation', 'question'])

    def fit(self, X, y=None):
        '''No fit required, defined to avoid errors'''
        return self
    
#     def fit_transform(self, X, y = None):
#         return self.transform(X)
    
    def _get_postag_count(self, data):
        nouns, adjectives, verbs, adverbs, sen_count, length = 0, 0, 0, 0, 0, 0       
        for sent in sent_tokenize(data):     # Break the document into sentences
            for token, tag in pos_tag(wordpunct_tokenize(sent)):     # Break the sentence into part of speech tagged tokens
                # Apply preprocessing to the token
                #token = token.lower()
                token = token.strip().strip('_').strip('*')               
                if tag.startswith("NN"):
                    nouns += 1
                elif tag.startswith("JJ"):
                    adjectives += 1
                elif tag.startswith("VB"):
                    verbs += 1
                elif tag.startswith("RB"):
                    adverbs += 1
                #else:
#                    print("Token : %s, Tag: %s" %(token,tag))
            sen_count += 1
        length = len(data)      
        nouns_p = nouns/length
        adjectives_p = adjectives/length
        verbs_p = verbs/length
        adverbs_p = adverbs/length
        
        #print("nouns: %d, adjectives: %d, verbs: %d, adverbs: %d, len: %d" % (nouns, adjectives, verbs, adverbs, l) )
        if length==0:
            print("Nasty Sentence ")
            return [0.,0.,0.,0.]
        else:
            return [nouns_p, adjectives_p, verbs_p, adverbs_p]
        
    def add_phrasal_feat_cnt(self, text):
        blob = TextBlob(text)
        cnt = len(blob.noun_phrases)
        return cnt

    def add_phrasal_feat_total(self, text):
        blob = TextBlob(text)
        t_cnt = sum(blob.np_counts.values())
        return t_cnt

    ## Words which are to be considered as feature
    def feature_words(self):
        word_list = ['escalation','terminate','liquidated','termination','fail','worksite','offshore', 'reasonable',
                     'heliport','transport','undisputed','changes','annual','adjustment', 'demobilisation']
        return word_list
    
    def word_count(self,text, word):
        return len(re.findall('[^\w+]'+ word + '[^\w+]' , text))

    #Syllable Count Lexicon Count Sentence Count
    def extract_structural_feature(self,data):
        ''' data is input pandas series of text'''
        feature_df = pd.DataFrame()
        feature_df["avg_sent_len"] =  data.apply(textstat.avg_sentence_length)
        feature_df["avg_sent_wrd"] =  data.apply(textstat.avg_sentence_per_word)  
        feature_df["syll_cnt"] =      data.apply(textstat.syllable_count)
        feature_df["lex_cnt"] =       data.apply(textstat.lexicon_count) 
        feature_df["sent_cnt"] =      data.apply(textstat.sentence_count) 
        feature_df["diff_word_cnt"] = data.apply(textstat.difficult_words) 
        feature_df["char_cnt"] =      data.apply(textstat.char_count)
        feature_df["unq_np_cnt"]   =  data.apply(self.add_phrasal_feat_cnt)
        #feature_df["total_np_cnt"] = data.apply(self.add_phrasal_feat_total)

        for word in self.feature_words():
            feature_df[word] = (data.apply(lambda x : self.word_count(x,word)))
        return feature_df


    def transform(self, documents):
        print('linguistic vector transform')
        nouns, adjectives, verbs, adverbs = np.array([self._get_postag_count(d) for d in documents]).T

        allcaps = []
        exclamation = []
        question = []
        for d in documents:
            try:
                allcaps.append(np.sum([t.isupper() for t in d.split() if len(t) > 2]))
                exclamation.append(d.count("!"))
                question.append(d.count("?"))
            except:
                print("documents are %s" %(d))
                continue

        result = np.array([nouns, adjectives, verbs, adverbs, allcaps, exclamation, question]).T
        result = pd.DataFrame(result)
        result.columns = ['count_'+ feature for feature in self.get_feature_names()]
        
        
        result1 = self.extract_structural_feature(documents)
        result = pd.concat([result, result1], axis =1)
        return result


from lightgbm import LGBMClassifier
class lgb_class(BaseEstimator, TransformerMixin):
    
    def __init__(self, n_estimators = 10000, params = {
        "objective" : "multiclass", 
        "num_class" : 7,
        "metric" : "multi_logloss",
        "learning_rate" : 0.05,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        #"num_leaves" : 30,
        "bagging_seed" : 2018,
        "verbosity" : -1
    } ):
        self.params = params
        self.n_estimators = n_estimators
        print('lgb_class')
    
    def fit(self, X, y= None):
        print('................lgb Fit...................')
        self.model = LGBMClassifier(**self.params)        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True)
        
        self.model.fit(X, y, eval_set  = [(X, y)], early_stopping_rounds = 1000, verbose =500)
#         self.model.fit(X_train, y_train, eval_set  = [(X_test, y_test)], early_stopping_rounds = 1000, verbose =500)
        return self
    
    def transform(self, X):
        print('................lgb transform...................')
        # No transformations needed
        return self
    
    def predict_proba(self, X):
        print('.................lgb predict_proba..................')
        return self.model.predict_proba(X)
    
    def predict(self, X):
        print('.................lgb predict..................')
        return self.model.predict(X)
    

from xgboost import XGBClassifier
class xgb_class(BaseEstimator, TransformerMixin):    
    def __init__(self, n_estimators = 10000, params = {'objective': 'multi:softprob', "num_class" : 17,
          'eval_metric': 'mlogloss',
          'eta': 0.05,
          'subsample': 0.7,
          'colsample_bytree': 0.5,
          'print.every.n': 10,
          'max_depth': 15, 
          'alpha': 0,
          'random_state': 42,
          'silent': True}):
        self.params = params
        self.n_estimators = n_estimators
        print('xgb_class')
    
    def fit(self, X, y= None):
        print('................xgb Fit...................')
        self.model = XGBClassifier(**self.params, n_estimators = self.n_estimators, early_stopping_rounds=10)
        
        self.model.fit(X, y,eval_set  = [(X, y)], early_stopping_rounds = 1000, verbose =500)
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True)        
#         self.model.fit(X_train, y_train, eval_set  = [(X_test, y_test)], early_stopping_rounds = 1000, verbose =500)
        return self
    
    def transform(self, X):
        print('................xgb transform...................')
        # No transformations needed
        return self
    
    def predict_proba(self, X):
        print('.................xgb predict_proba..................')
        return self.model.predict_proba(X)
    
    def predict(self, X):
        print('.................xgb predict..................')
        return self.model.predict(X)
    


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

class custom_vectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model = 'tfidf'):
        self.model = model
        print('custom_vectorizer')
    
    print('in custom_vectorizer outside init')
    def fit(self, X, y = None):
        # no fitting required
        return self
    
#     def fit_transform(self, X, y = None):
#         print('custom_vectorizer transform....')
#         return self.transform(X)
    
    def transform(self, X):
        print(" in custom transform")
        X_features = pd.DataFrame()
        X_features['vect_sum'] = list(X.sum(axis =1).flat)
        X_features['vect_mean'] = X.mean(axis=1)
        X_features['vect_len'] =  (X != 0).sum(axis = 1)
        
        X_features.columns = [col+'_'+self.model for col in X_features.columns ]
        print("Exiting custom_vectorizer")
        return X_features


    
class DictionaryBased_Preprocessor(BaseEstimator, TransformerMixin):
    """    Transforms input data by using Dictionary of words and regular expressions.    """
    def __init__(self, dictionary = None):
        self.dictionary = dictionary
    
    def fit(self, X, y=None):
        '''No fit required, defined to avoid errors'''
        return self
    
    def transform(self, X):
        print('dict transform')
        for i in range(len(X)):
            for k, v in self.dictionary.items():
                X[i] = re.sub(k, v, ' '.join(X[i]));
        print(i)
        return X
    


from spellchecker import SpellChecker
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from nltk import sent_tokenize
from nltk import WordNetLemmatizer
import string
import wordninja

class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.    """
    def __init__(self, lower=True, strip=True):
        """        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.        """
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(sw.words('english'))
        self.punct      = set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.spell      = SpellChecker()

    def fit(self, X, y=None):
        print('NLTK fit....')
        """        Fit simply returns self, no other information is needed.        """
        return self
    
#     def fit_transform(self, X, y = None):
#         return self.transform(X)
    
    def transform(self, X):
        """        Actually runs the preprocessing on each document.        """
        print('NLTK transform....')
        l=[]
        for doc in X:
            doc = ' '.join(wordninja.split(doc))
            sent = list(self.tokenize(doc))            
            misspelled = self.spell.unknown((sent))
            new_words = [word for word in sent if word not in misspelled]
            l.append(new_words)
        return l

    def tokenize(self, document):
        """        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.        """
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip().strip('_').strip('*') if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)
    
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline


def preprocess_text(X):
    preprocess_text = FeatureUnion([
        ('ling_st', LinguisticVectorizer()),
        ('processed_text_pipe', Pipeline([
            ('nltk_preprocessor', NLTKPreprocessor()),
            ('dict_process', DictionaryBased_Preprocessor(re_repl)),
            ('processed_text_feature', FeatureUnion([
                ('vecCustom_cvec', Pipeline([
                    ('cvec', CountVectorizer(min_df = 5, ngram_range= (1,2), max_features = 500,
                                        strip_accents='unicode', token_pattern=r'\w+', stop_words = 'english')),
                    ('vecCustom', custom_vectorizer(model = 'count'))
                    ])
                ),
                ('tfidf_pipe', Pipeline([
    #                 max_features = 500,
                    ('tfidf', TfidfVectorizer(min_df = 3, ngram_range= (1,2), sublinear_tf=1,
                                        strip_accents='unicode', token_pattern=r'\w+', stop_words = 'english')),
                    ('tfidf_process', FeatureUnion([
                        ('svd_scale', Pipeline([
                            ('svd', TruncatedSVD(n_components = 120, algorithm = 'arpack')),
                            ('scale', StandardScaler())
                            ])
                        ),
                        ('vecCustom_tfidf', custom_vectorizer(model = 'tfidf1'))
                        ])
                    )
                    ])
                )
                ])
            )
            ])
        )
        ])
    return preprocess_text.fit_transform(X)

class final_fit(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('final_fit')
        
    def fit(self, X, y= None):
        self.svc = SVC(C=1.0, probability=True)
        self.xgb_clf = xgb_class(n_estimators= 100)
        self.lgb_clf = lgb_class(n_estimators= 100)
        
        print(X.shape)
        self.svc.fit(X[:, 30:], y)
        self.xgb_clf.fit(X, y)
        self.lgb_clf.fit(X, y)
        
        print('............fitting done...............')
        
        return self
    
    def transform(self, X):
        return self
        
    def predict(self, X):        
        svc_predict = self.svc.predict(X[:, 30:])
        xgb_predict = self.xgb_clf.predict(X)
        lgb_predict = self.lgb_clf.predict(X)
        return [xgb_predict, lgb_predict]
    
    def predict_proba(self, X):
        svc_predict = self.svc.predict_proba(X[:, 30:])
        xgb_predict = self.xgb_clf.predict_proba(X)
        lgb_predict = self.lgb_clf.predict_proba(X)
        return [xgb_predict, lgb_predict]
    
    
from gensim.models import keyedvectors as word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
from tqdm import tqdm



# def load_embeddings_from_txt(EMBEDDING_FILE):
#     embeddings_index = dict()
#     f = open(EMBEDDING_FILE, encoding="utf-8")
#     #Transfer the embedding weights into a dictionary by iterating through every line of the file.
#     for line in tqdm(f):
#         values = line.split()
#         try:
#             word = values[0]
#             coefs = np.asarray(values[1:], dtype='float32')
#             embeddings_index[word] = coefs
#         except ValueError:
#             pass
#     f.close()
#     return embeddings_index


# def load_embedding_index(filepath, embeddings_type = 'glove'):
#     #load different embedding file from Kaggle depending on which embedding matrix we are going to experiment with
#     embeddings_index = dict()

# #     if(embeddings_type in ["glove", "conceptnet"]):
# #         EMBEDDING_FILE = filepath
# #         embeddings_index = load_embeddings_from_txt(EMBEDDING_FILE)

# #     elif(embeddings_type=="conceptnet"):
# #         EMBEDDING_FILE = filepath
# #         embeddings_index = load_embeddings_from_txt(EMBEDDING_FILE)

#     if(embeddings_type=="fasttext"):
#         EMBEDDING_FILE = filepath
#         fasttextDict = word2vec.KeyedVectors.load_word2vec_format(fasttext_data_file, binary=False, encoding='utf8')
#         for word in fasttextDict.wv.vocab:
#             embeddings_index[word] = fasttextDict.word_vec(word) 
#     else:
#         EMBEDDING_FILE = filepath
#         embeddings_index = load_embeddings_from_txt(EMBEDDING_FILE)
        
# #     else:
# #         glove_file = datapath(filepath)
# #         tmp_file = get_tmpfile("word2vec.txt")
# #         _ = glove2word2vec(glove_file, tmp_file)            
# #         word2vecDict = word2vec.KeyedVectors.load_word2vec_format(tmp_file)
# #         for word in word2vecDict.wv.vocab:
# #             embeddings_index[word] = word2vecDict.word_vec(word)

#     print('Loaded %s word vectors.' % len(embeddings_index))
#     return embeddings_index

# # glove_data_file
# embeddings_index = load(conceptNet_data_file)
# #embeddings_index = load_embedding_index(conceptNet_data_file, embeddings_type = 'conceptNet')
# print("....................Loaded word embeddings........... ")


# def sent2vec(sent, embeddings_index, dim):
#     words = str(sent).lower() #.decode('utf-8')
#     words = word_tokenize(words)
#     words = [w for w in words if ((w not in stop_words) & (w.isalpha()))]
#     M = []
#     for w in words:
#         try:
#             M.append(embeddings_index[w])
#         except:
#             continue
#     M = np.array(M)
#     v = M.sum(axis=0)
#     if type(v) != np.ndarray:
#         return np.zeros(dim)
#     return v / np.sqrt((v ** 2).sum())


# class embedding_scaler(BaseEstimator, TransformerMixin):
#     def __init__(self, dim = 300):
#         self.dim = dim
#         print('embedding_scaler')
        
#     def fit(self, X, y =None):
#         self.scale_embedding = StandardScaler()
        
#         X_embeddings = np.array([sent2vec(sent, embeddings_index, self.dim) for sent in X])
#         X_embeddings_scaled = self.scale_embedding.fit(X_embeddings)
#         return self
    
#     def transform(self, X):
#         X_embeddings = np.array([sent2vec(sent, embeddings_index, self.dim) for sent in X])
#         X_embeddings_scaled = self.scale_embedding.transform(X_embeddings)
        
#         return X_embeddings_scaled
    

# from keras.models import Sequential
# from keras.layers import Activation, Dropout, Dense
# from keras.layers.normalization import BatchNormalization
# from keras.utils import np_utils
# from nltk import word_tokenize
# from nltk.corpus import stopwords
# stop_words = stopwords.words('english')

# class simple_seq(BaseEstimator, TransformerMixin):
    
#     def __init__(self, dim = 300, num_class = 7, opt='rmsprop'):
#         self.dim = dim
#         self.num_class = num_class
#         self.opt = opt
    
#     def fit(self, X, y= None):
#         y = np_utils.to_categorical(y)
        
#         self.model = Sequential()
#         self.model.add(Dense(self.dim, input_dim=self.dim, activation='relu'))
#         self.model.add(Dropout(0.2))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(self.dim, activation='relu'))
#         self.model.add(Dropout(0.3))
#         self.model.add(BatchNormalization())
#         self.model.add(Dense(self.num_class))
#         self.model.add(Activation('softmax'))
#         self.model.compile(loss='categorical_crossentropy', optimizer= self.opt)
        
#         # Fit the model with early stopping callback
#         #earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=30, verbose=0, mode='auto')
        
#         X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, shuffle=True)
        
#         self.model.fit(X_train, y_train, batch_size=16, epochs=50, verbose=1,
#                        validation_data=(X_test, y_test)) #, callbacks=[earlystop])
        
#         return self
    
#     def predict_proba(self, X):
#         return self.model.predict_proba(X)
    
#     def predict(self, X):
#         return self.model.predict(X).argmax(axis=1)
    
    
# class glove_run_xgb(BaseEstimator, TransformerMixin):
#     def __init__(self, dim=300, params = {'objective': 'multi:softprob', 
#           'eval_metric': 'mlogloss',
#           'print.every.n': 10,
#           'eta': 0.05,
#           'max_depth': 15, 
#           'subsample': 0.7, 
#           'colsample_bytree': 0.5,
#           'alpha':0,
#           'random_state': 42,
#           "num_class" : 7,
#           'silent': True,
#           'nthread':10}):
#         self.params = params
#         self.dim = dim
        
#     def fit(self, X, y):
#         self.model = XGBClassifier(**self.params)
#         self.model.fit(X, y)
        
#         return self
    
#     def predict_proba(self, X):
#         return self.model.predict_proba(X)
    
#     def predict(self, X):
#         return self.model.predict(X)
    

# class embeddings_models(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         print('embeddings_models')
        
#     def fit(self, X, y = None):
#         self.sequence = simple_seq()
#         self.xgb = glove_run_xgb()
        
#         self.sequence.fit(X,y)
#         self.xgb.fit(X, y)
#         return self
    
#     def predict(self, X):
#         return [self.sequence.predict(X), self.xgb.predict(X)]
    
#     def predict_proba(self, X):
#         return [self.sequence.predict_proba(X), self.xgb.predict_proba(X)]
    
    
# class assembly(BaseEstimator, TransformerMixin):
    
#     def __init__(self):
#         print('does nothing')
        
#     def fit(self, X, y):
#         self.preprocess_text = preprocess_text
#         self.clf = final_fit()
#         self.embedding_scaler = embedding_scaler()
#         self.embeddings_models = embeddings_models()
        
#         pipe_1 = self.preprocess_text.fit_transform(X)
#         pipe_2 = self.embedding_scaler.fit_transform(X)
        
#         self.clf.fit(pipe_1,y)
#         self.embeddings_models.fit(pipe_2,y)
#         return self
    
#     def transform (self,X, y = None):
#         pipe_1 = self.preprocess_text.transform(X)
#         pipe_2 = self.embedding_scaler.transform(X)
        
#         return pipe_1, pipe_2
    
#     def predict(self, X):
#         pipe_1, pipe_2 = self.transform(X)
#         out_predict = self.clf.predict(pipe_1) + self.embeddings_models.predict(pipe_2)
#         return out_predict
        
#     def predict_proba(self, X):
#         pipe_1, pipe_2 = self.transform(X)
#         x = self.clf.predict_proba(pipe_1)
#         print(len(x))
#         out_predict = [np.mean(x, axis =0),
#                        np.mean(self.embeddings_models.predict_proba(pipe_2), axis =0)
#                       ]
#         return out_predict


# class sub_clf(BaseEstimator, TransformerMixin):
    
#     def __init__(self):
#         print('Sub Classifier')
        
#     def fit(self, X, y):
#         self.embedding_scaler = embedding_scaler()
#         pipe_2 = self.embedding_scaler.fit_transform(X)
        
#         self.sub_clf = glove_run_xgb()
#         self.sub_clf.fit(pipe_2, y)
#         return self
    
#     def transform (self,X, y = None):
#         pipe_2 = self.embedding_scaler.transform(X)
#         return pipe_2
    
#     def predict(self, X):
#         pipe_2 = self.transform(X)
#         out_predict = self.sub_clf.predict(pipe_2)
#         return out_predict
        
#     def predict_proba(self, X):
#         pipe_2 = self.transform(X)
#         out_predict = self.sub_clf.predict_proba(pipe_2)
#         return out_predict