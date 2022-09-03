import os
import sys
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


pd.set_option('display.max_colwidth', 120)
pd.set_option('display.max_rows', 100)


dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from classes import preprocess_text
from functions import get_dataframe, cm_analysis
from functions import NLTKPreprocessor, Best_estimator_grid

n_components  =  sys.argv[1]
gmm_model_name = 'GMM_Components_'+str(n_components) + '.pickle'
file_test = os.path.join(dirname,'labelled_dataset.csv')

fig_name = 'confusion_matrix_'+ gmm_model_name.split('.')[0] + '.png'
supervised_model = 'Supervised_svm_'+ gmm_model_name.split('.')[0] + '.joblib'

df = get_dataframe(file_test)

labels = LabelEncoder()
labels.fit(df['Clause'].unique().tolist())
print("Label Encoding Classes are : ")
print(labels.classes_)
# df = df.iloc[0:200]
print("Preprocessing supervised data ........")
X_super = preprocess_text(df['Text'].astype(str))


GMM_model = load(gmm_model_name)
feature = GMM_model.predict_proba(X_super)

X = np.asarray(feature)
Y = np.asarray(labels.transform(df['Clause']))
print(np.shape(X), np.shape(Y))
X, Y = shuffle(X, Y)
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.2, random_state=42)
clf = SVC()
params = {'C': [1, 10, 100, 1000, 1500],
           'kernel' : [ 'rbf'],
           'gamma' : ['auto']     }

print("Finding optimal SVM model ............")

optimal_model = Best_estimator_grid(X_train, y_train, clf, params, 5)
# optimal_model = load(supervised_model)
print("Optimal Model of SVM for K means is" , optimal_model.best_estimator_)
best_model = optimal_model.best_estimator_
best_model.fit(X_train, y_train)

y_pred = best_model.predict(X_test)
inv_map = {int(k): v for k, v in enumerate(labels.classes_)}
figure = cm_analysis(y_test, y_pred, labels = list(range(len(inv_map))), ymap = inv_map, figsize=(15,15))
figure.tight_layout()
figure.savefig(fig_name)
dump(optimal_model, supervised_model)



