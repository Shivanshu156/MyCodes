import os, sys
import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from math import sqrt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import numpy as np
from joblib import load, dump


dirname = os.path.dirname(__file__)
# dirname = '/Users/shivanshu/Desktop/Sem IX/MTP'
sys.path.insert(0, dirname)
filename = os.path.join(dirname, 'featured_data_111_C.csv')
GBR_model_name = os.path.join(dirname, 'GBR_C_111.pickle')


data = pd.read_csv(filename)
data = data.drop(['Unnamed: 0'],axis=1)
Y = data.loc[:,'reactionEnergy'].values
# X = data.loc[:, 'AN' : 'dBandCenter'].values

X = data.loc[:, ['M.P.','B.P.','H_FUS','IE', "SE","M.P.'","B.P.'","H_FUS'","IE'", "SE'","ElectronAffinity", "Pauling", "WorkFunction", "dBandCenter" ]]
X = data.loc[:, ['EN',"EN'" ]].values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

def Best_estimator_grid(X, Y, clf, params, cv):
    """ Returns the best estimator from a dictionary of parameters"""
    print("Within Best_estimator")
    clf_grid = GridSearchCV(clf, params, cv=cv, n_jobs=2, verbose=10)
    clf_grid.fit(X, Y)
    return clf_grid


clf = GradientBoostingRegressor()
# params = {'n_estimators': [ 10, 100],
#            'learning_rate':[ .1],
#            'loss':['ls']  }
params = {'n_estimators': [ 10, 200, 1000, 500],
           'learning_rate':[.1, 0.05, 1],
           'loss':['ls', 'lad', 'huber'], 'max_depth' : [ 3, 4],
        #    'max_features' : [10, 14]
            }

print("Finding optimal GBR model ............")

optimal_model = Best_estimator_grid(X_train, y_train, clf, params, 5)
best_model = optimal_model.best_estimator_
print("Optimal Model of GBR is" , best_model)
dump(best_model, os.path.join(dirname, GBR_model_name))
print("Model has been dumped Successfully !!!")
print("Finding root mean square error ............")

errors=[]
errors_train=[]
feature_importances_array=np.zeros(14)
error_list=[]
error_all_i=np.zeros(14)
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
