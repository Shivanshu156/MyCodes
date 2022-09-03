import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import pos_tag
from nltk import sent_tokenize
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from nltk import WordNetLemmatizer
from nltk import wordpunct_tokenize
from spellchecker import SpellChecker
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords as sw
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.base import BaseEstimator, TransformerMixin


def get_dataframe(input_file_path, sheet = None):
    """
    Takes input as excel or csv file, returns dataframe
    """
    print("In function get_dataframe")
    file_type = input_file_path.split('.')[-1]
    print("file type is" + file_type)
    if file_type == 'csv':
        # print("in csv")
        corpus = pd.read_csv(input_file_path)
    elif file_type is 'xls' or 'xlsx':
        if sheet:
            corpus = pd.read_excel(input_file_path, sheet_name=sheet)
        else:
            corpus = pd.read_excel(input_file_path)
    return corpus


def preprocess_dataframe(df, similarity):
    df = df.drop_duplicates(subset='Text', keep ='first')
    df.sort_values(['Clause'], inplace=True)
    df.reset_index(inplace=True)
    df1 = df
    size = 0
    for clause, df_clause in df.groupby('Clause'):
        # df_clause.reset_index(inplace=True)
        size += df_clause.shape[0]
        print('size is ', size)
        for i, row in df_clause.iterrows():
            for j in range(i+1,size):
                # print(i, j)
                if SequenceMatcher(None, df['Text'].iloc[i], df['Text'].iloc[j]).ratio() > similarity:
                        df1['Text'].iloc[j] = '' 
    df1 = df1[df1.Text != '']
    return df1        



def Best_estimator_grid(X, Y, clf, params, cv):
    """ Returns the best estimator from a dictionary of parameters"""
    print("Within Best_estimator")
    clf_grid = GridSearchCV(clf, params, cv=cv, n_jobs=2, verbose=10)
    clf_grid.fit(X, Y)
    return clf_grid
    

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(5,5)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if np.isnan(p):
                p = 0
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
            cm[i,j] = p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    acc = accuracy_score(y_true, y_pred)*100
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted  : Accuracy = '+ str(acc) + '%'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax,cmap='Blues')
    plt.show()
    return fig


class NLTKPreprocessor(BaseEstimator, TransformerMixin):
    """
    Transforms input data by using NLTK tokenization, lemmatization, and
    other normalization and filtering techniques.
    """

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        """        Instantiates the preprocessor, which make load corpora, models, or do
        other time-intenstive NLTK data loading.        """
        self.lower = lower
        self.strip = strip
        self.stopwords = set(stopwords) if stopwords else set(sw.words('english'))
        self.punct = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()

    def fit(self, X, y=None):
        """        Fit simply returns self, no other information is needed.        """
        return self

    def inverse_transform(self, X):
        """        No inverse transformation        """
        return X

    def transform(self, X):
        """        Actually runs the preprocessing on each document.        """
        l = []
        #         print('NLTK preprocessing...')
        #         print(type(X))
        for doc in X:
            #             print("Here we are")
            sent = list(self.tokenize(doc.lower()))
            #             misspelled = self.spell.unknown((sent))
            #             new_words = [word for word in sent if (word not in misspelled) & (word.isalpha())]
            new_words = [word for word in sent if word.isalpha()]

            # Remove 'BSO' and domain name from the description
            l.append(' '.join(new_words))
        return l

    def tokenize(self, document):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """

        document = re.sub(r'((fw|re):|\((.*?)\)|\[(.*?)\]|wrqst)', '', document)
        document = re.sub(r' +', ' ', document)
        # Break the document into sentences
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

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


