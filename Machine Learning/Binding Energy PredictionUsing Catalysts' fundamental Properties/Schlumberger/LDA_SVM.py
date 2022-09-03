import os, re
import sys
import gensim
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

dirname = os.path.dirname(__file__)
#dirname = r'C:\Users\ABansal4\OneDrive - Schlumberger\CRT\Contract-Analysis'
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis
from functions import NLTKPreprocessor, Best_estimator_grid

max_f = int(sys.argv[1])
n = int(sys.argv[2])
process_ppl_name = 'preprocess_ppl_' + str(max_f) + '.pickle'
lda_model_name = 'lda_model_' + str(max_f) + '_features.pickle'
csv_file_name = 'classification_report_' + lda_model_name.split('.')[0] + str(n) + '% others' + '.csv'
fig_name = 'confusion_matrix_' + lda_model_name.split('.')[0] + str(n) + '% others' + '.png'
supervised_model = 'Supervised_svm'  + lda_model_name.split('.')[0] + str(n) + '% others' + '.joblib'
# file_test = os.path.join(dirname,'Tagged_Dataset.xlsx')
file_test = os.path.join(dirname,'labelled_dataset_with_others.csv')
df = get_dataframe(file_test)
# df_others = df[df.Clause == 'Others']
# df_clause = df[df.Clause != 'Others']
# df_others = shuffle(df_others)
# df_others = df_others.iloc[0:int(n*len(df_others)/100)]

# df = pd.concat([df_clause,df_others], ignore_index=True)
# df = shuffle(df)

c_list = ['Refusal', 'Standby', 'Payment Terms', 'Price Adjustments']
df = df[~df.Clause.isin(c_list)]

ldamodel = load(os.path.join(dirname, lda_model_name))
# ldamodel.print_topics(num_topics=num_topics, num_words=10)
print("LDA Model Loaded Successfully !!!")

preprocess_pipeline = load(os.path.join(dirname, process_ppl_name))
print("Pipeline Loaded")
def topic_distribution(new_doc):
    new_doc_corpus = gensim.matutils.Sparse2Corpus(new_doc, documents_columns=False)
    doc_topics = ldamodel.get_document_topics(new_doc_corpus)
    return list(doc_topics)

data = preprocess_pipeline.transform(df['Text'].astype(str))
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
# X, Y = shuffle(X, Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, stratify = Y,test_size=0.2, shuffle=True)

clf = SVC()
params = {'C': [ 10, 800,900, 1000, 1500],
           'kernel' : ['linear', 'rbf'],
           'gamma' : ['auto']     }

print("Finding optimal SVM model ............")

optimal_model = Best_estimator_grid(X_train, y_train, clf, params, 5)
# optimal_model = load(supervised_model)
print("Optimal Model of SVM is" , optimal_model)
best_model = optimal_model.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
inv_map = {int(k): v for k, v in enumerate(labels.classes_)}
figure = cm_analysis(y_test, y_pred, labels = list(range(len(inv_map))), ymap = inv_map, figsize=(15,15))
figure.tight_layout()
figure.savefig(fig_name)
dump(optimal_model, supervised_model)


def classification_report_csv(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=2)
    print(report)
    report_data = []
    lines = report.split('\n')
    for line in lines[2:]:
        row = {}
#        line = re.sub('  +', '  ', line)
        row_data = re.split('  +', line)
        row_data = list(filter(lambda x: x!='', row_data))
        try:
            row['class'] = row_data[0]
            row['precision'] = (row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
        except:
            row = {'class':'', 'precision': '', 'recall':'', 'f1_score':'', 'support': ''}
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)    
    return dataframe  

report = classification_report_csv(y_test, y_pred)
report.to_csv(os.path.join(dirname, csv_file_name))

