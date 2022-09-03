import os, sys
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.svm import SVR
from joblib import load, dump
from sklearn import model_selection
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor


dirname = os.path.dirname(__file__)
# dirname = '/Users/shivanshu/Desktop/Sem IX/MTP'
sys.path.insert(0, dirname)
filename = os.path.join(dirname, 'featured_data_111_C.csv')
GBR_model_name = os.path.join(dirname, 'GBR_C_111.pickle')


data = pd.read_csv(filename)
data = data.drop(['Unnamed: 0'],axis=1)
Y = data.loc[:,'reactionEnergy'].values
X = data.loc[:, 'AN' : 'dBandCenter'].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

def Best_estimator_grid(X, Y, clf, params, cv):
    """ Returns the best estimator from a dictionary of parameters"""
    print("Within Best_estimator")
    clf_grid = GridSearchCV(clf, params, cv=cv, n_jobs=2, verbose=10)
    clf_grid.fit(X, Y)
    return clf_grid

#############################################
clf = GradientBoostingRegressor()
# params = {'n_estimators': [ 10, 100],
#            'learning_rate':[ .1],
#            'loss':['ls']  }
params = {'n_estimators': [ 10, 100, 200, 500, 1000],
           'learning_rate':[.1, 1, .01],
           'loss':['ls', 'lad', 'huber'], 'max_depth' : [2, 3, 4],
           'max_features' : [10, 15, 20, 28] }

print("Finding optimal GBR model ............")

optimal_model = Best_estimator_grid(X_train, y_train, clf, params, 5)
#############################################

best_model = LinearRegression(n_jobs=3)
best_model.fit(X_train, y_train)
y_train_predicted=best_model.predict(X_train)
y_test_predicted=best_model.predict(X_test)

#############################################

best_model = optimal_model.best_estimator_
print("Optimal Model of GBR is" , best_model)
dump(best_model, os.path.join(dirname, GBR_model_name))
print("Model has been dumped Successfully !!!")
print("Finding root mean square error ............")

errors=[]
errors_train=[]
feature_importances_array=np.zeros(28)
error_list=[]
error_all_i=np.zeros(28)
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    best_model.fit(X_train, y_train)
    y_train_predicted=best_model.predict(X_train)
    y_test_predicted=best_model.predict(X_test)

    # Later find which values create most errors
    errors.append(sqrt(mean_squared_error(y_test, y_test_predicted)))
    errors_train.append(sqrt(mean_squared_error(y_train, y_train_predicted)))
    b=best_model.feature_importances_
    feature_importances_array=np.add(feature_importances_array,b)
    i=i+1
    print(i)

print ("Train Error" + str(sum(errors_train)/float(len(errors_train)))) 
print ("Test Error" + str(sum(errors)/float(len(errors))))
final_importance=np.divide(feature_importances_array,100.0)

with open('GBR_C_111_results.txt', 'w') as f:
    print('Test_error:', str(sum(errors)/float(len(errors))), file=f)  # Python 3.x
    print('Train_error:', str(sum(errors_train)/float(len(errors_train))), file=f)  # Python 3.x

#######################################################################################################


from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# df_others = df[df.Clause == 'Others']
# df_clause = df[df.Clause != 'Others']
# df_others = shuffle(df_others)
# df_others = df_others.iloc[0:int(n*len(df_others)/100)]

# df = shuffle(df)

c_list = ['Refusal', 'Standby', 'Payment Terms', 'Price Adjustments']
df = df[~df.Clause.isin(c_list)]


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


report.to_csv(os.path.join(dirname, csv_file_name))