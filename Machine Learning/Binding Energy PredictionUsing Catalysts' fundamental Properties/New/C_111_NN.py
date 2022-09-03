import os, sys
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.svm import SVR
from joblib import load, dump
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor


dirname = os.path.dirname(__file__)
# dirname = '/Users/shivanshu/Desktop/Sem IX/MTP'
sys.path.insert(0, dirname)
filename = os.path.join(dirname, 'Featured_data_C_111.xlsx')
NN_model_name = os.path.join(dirname, 'NN_C_111.pickle')


data = pd.read_excel(filename)
# data = data.drop(['Pauling_x','WorkFunction_x','dBandCenter_x'],axis=1)
# data = data.drop(['Pauling_y','WorkFunction_y','dBandCenter_y'],axis=1)
data.dropna(inplace=True)
Y = data.loc[:,'Enegry'].values
# X = data.loc[:, "AN'_x" : "SE'_y"].values
X = data.loc[:, "AN'_x" : "dBandCenter_y"].values

# X = data.loc[:, ['M.P.','B.P.','H_FUS','IE', "SE","M.P.'","B.P.'","H_FUS'","IE'", "SE'","ElectronAffinity", "Pauling", "WorkFunction", "dBandCenter" ]]
# X = data.loc[:, ['EN',"EN'" ]].values
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

def Best_estimator_grid(X, Y, clf, params, cv):
    """ Returns the best estimator from a dictionary of parameters"""
    print("Within Best_estimator")
    clf_grid = GridSearchCV(clf, params, cv=cv, n_jobs=2, verbose=10)
    clf_grid.fit(X, Y)
    return clf_grid


clf = MLPRegressor()
params = {
    'hidden_layer_sizes': [  (200,200,200,200) ],
           'activation':[  'tanh'],
        #    'solver':['adam'], 'learning_rate' :
        #    ['adaptive']
            }
params = {'hidden_layer_sizes': [  (10, 20, 10), (10,20), (10,20,5)],
           'activation':['identity', 'logistic', 'tanh','relu'],
           'solver':['lbfgs', 'sgd', 'adam'], 'learning_rate' :
           ['constant', 'invscaling', 'adaptive'] }

print("Finding optimal Neural Network model ............")

optimal_model = Best_estimator_grid(X_train, y_train, clf, params, 5)
best_model = optimal_model.best_estimator_
print("Optimal Model of GBR is" , best_model)
dump(best_model, os.path.join(dirname, NN_model_name))
print("Model has been dumped Successfully !!!")
print("Finding root mean square error ............")

errors=[]
errors_train=[]
feature_importances_array=np.zeros(X.shape[1])
error_list=[]
error_all_i=np.zeros(X.shape[1])
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
    best_model.fit(X_train, y_train)
    y_train_predicted=best_model.predict(X_train)
    y_test_predicted=best_model.predict(X_test)

    # Later find which values create most errors
    errors.append(sqrt(mean_squared_error(y_test, y_test_predicted)))
    errors_train.append(sqrt(mean_squared_error(y_train, y_train_predicted)))
    # b=best_model.feature_importances_
    # feature_importances_array=np.add(feature_importances_array,b)
    i=i+1
    print(i)

print ("Train Error" + str(sum(errors_train)/float(len(errors_train)))) 
print ("Test Error" + str(sum(errors)/float(len(errors))))
# final_importance=np.divide(feature_importances_array,100.0)

with open('NN_C_111_results.txt', 'w') as f:
    print('Test_error:', str(sum(errors)/float(len(errors))), file=f)  # Python 3.x
    print('Train_error:', str(sum(errors_train)/float(len(errors_train))), file=f)  # Python 3.x
    print('R square score for test data is ',str(best_model.score(X_test, y_test)), file=f)
    print('R square score for train data is ',str(best_model.score(X_train, y_train)), file=f)
