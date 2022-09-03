import os
import gensim
import numpy as np
import pandas as pd
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
from functions import get_dataframe, cm_analysis, NLTKPreprocessor


pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_rows', 100)

#file_train = r"C:\Users\sverma9\Desktop\Shivanshu\LDA\input_train.txt"
#text = open(file_train, 'r').read()
#
#corpus = pd.DataFrame({'text': text.split('\n')})
# print(corpus.loc[[0]])
# tfidf = models.TfidfModel(corpus)

# dirname = os.path.dirname(__file__)
dirname = r'C:\Users\sverma9\Desktop\Shivanshu\LDA_final'
file_train = os.path.join(dirname,'complete_dataset.csv')
file_test = os.path.join(dirname,'New Clauses_ SLB Examples for Quickstudies_20190301_143554_182.xls')
corpus = get_dataframe(file_train)
df = get_dataframe(file_test, sheet='New Clauses  SLB Examples for Q')
df.rename(columns={'Result': 'Text', 'Field Name': 'Clause'}, inplace=True)

sw1 = stopwords.words("english")
sw1.extend(['shall', 'company', 'new', 'saudi', 'contract', 'contractor', 'must', 'good', 'subcontractor', 'may',
            'schlumberger'])

num_topics = 100
ngram_range = (1, 2)

# np.set_printoptions(precision=2)
# ################################
#Pre_processing_pipeline = Pipeline([
#        ('pre_processing', NLTKPreprocessor()),
#        ('vect', CountVectorizer(max_df = .5, analyzer='word', ngram_range = ngram_range))
#        ])
#        
#text_process = Pre_processing_pipeline.fit_transform(corpus['Text'])
    
    ##############################

ppl = Pipeline([('nltk',NLTKPreprocessor()),('vect', CountVectorizer(max_df=.5, analyzer='word', ngram_range=ngram_range))  ] )
X = ppl.fit_transform(corpus['Paragraph'])
id_map = dict((v, k) for k, v in ppl.named_steps.vect.vocabulary_.items())
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)
# tfidf = gensim.models.TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=id_map, passes=40, random_state=34)
print("LDA model has been trained Successfully !!!")
# ldamodel.print_topics(num_topics=num_topics, num_words=10)


def topic_distribution(new_doc):

    new_doc_corpus = gensim.matutils.Sparse2Corpus(new_doc, documents_columns=False)
    doc_topics = ldamodel.get_document_topics(new_doc_corpus)
    return list(doc_topics)

#df = pd.read_excel(r'C:\Users\sverma9\Desktop\Shivanshu\LDA\Contract_Clause_Dataset_Interns.xlsx', sheet_name=list(range(0, 6)))
#df1 = pd.DataFrame()
#for k, v in df.items():
#    df1 = pd.concat([df1, v], sort=True)
#df1.dropna(subset=['Text'], inplace=True)
#df1['Output Cluster'] = ''
#df1['Output Cluster Probability'] = ''


data = ppl.transform(df['Text'])
top_dist = topic_distribution(data)


labels = LabelEncoder()
labels.fit(df['Clause'].unique().tolist())
print(labels.classes_)


features = []
for topic in top_dist:
    feature = np.zeros(100).tolist()
    for item in topic:
        feature[item[0]] = item[1]
    features.append(feature)

# Supervised Model
X = np.asarray(features)
Y = np.asarray(labels.transform(df['Clause']))
print(np.shape(X), np.shape(Y))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42, shuffle=True)

# clf_SVC = SVC(C=100, gamma='auto')
# clf_SVC.fit(X_train, Y_train)
# Y_pred = clf_SVC.predict(X_test)
# plot_confusion_matrix(Y_test, Y_pred, title='Confusion matrix, without normalization')


# Grid Search CV
params = {'C': [1, 10, 100, 800, 1000, 1500, 10000, 100000]}
clf = SVC(gamma='auto')
clf_grid = GridSearchCV(clf, params, cv=5)
clf_grid.fit(X, Y)

print(clf_grid.cv_results_)
print(clf_grid.best_estimator_)

y_pred_grid = clf_grid.best_estimator_.predict(X)

inv_map = {int(k): v for k, v in enumerate(labels.classes_)}

cm_analysis(Y, y_pred_grid, labels = list(range(len(inv_map))), ymap = inv_map, figsize=(10,10))









