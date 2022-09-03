import os, sys
import numpy as np
import pandas as pd
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from joblib import load, dump
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


dirname = os.path.dirname(__file__)
# dirname = '/Users/shivanshu/Desktop/Sem IX/MTP'
sys.path.insert(0, dirname)
# filename = os.path.join(dirname, 'Featured_data_C_111.xlsx')
filename = os.path.join(dirname, 'Featured_data_C_111_refined.xlsx')
GBR_model_name = os.path.join(dirname, 'GBR_C_111_new.pickle')


data = pd.read_excel(filename)

# data = data.drop(['Pauling_x','WorkFunction_x','dBandCenter_x'],axis=1)
# data = data.drop(['Pauling_y','WorkFunction_y','dBandCenter_y'],axis=1)
data.dropna(inplace=True)
Y = data.loc[:,'Enegry'].values
# X = data.loc[:, "AN'_x" : "SE'_y"].values
X = data.loc[:, "AN'_x" : "dBandCenter_y"].values

# X = data.loc[:, ['M.P.','B.P.','H_FUS','IE', "SE","M.P.'","B.P.'","H_FUS'","IE'", "SE'","ElectronAffinity", "Pauling", "WorkFunction", "dBandCenter" ]]
# X = data.loc[:, ['EN',"EN'" ]].values

# scaler = StandardScaler()
# scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)

def Best_estimator_grid(X, Y, clf, params, cv):
    """ Returns the best estimator from a dictionary of parameters"""
    print("Within Best_estimator")
    clf_grid = GridSearchCV(clf, params, cv=cv, n_jobs=8, verbose=1)
    clf_grid.fit(X, Y)
    return clf_grid


clf = GradientBoostingRegressor()
# params = {'n_estimators': [ 10, 100],
#            'learning_rate':[ .1],
#            'loss':['ls']  }
params = {'n_estimators': [  100, 80, 50],
           'learning_rate':[.1, 0.075, 0.125],
           'loss':['lad', 'huber', 'quantile'], 'max_depth' : [ 2, 3, 4],
        #    'max_features' : [11, 12, 13, 14]
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
feature_importances_array=np.zeros(X_train.shape[1])
error_list=[]
error_all_i=np.zeros(X_train.shape[1])
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
feature_name  = ["AN'_x", "AM'_x", "G'_x", "P'_x",
       "R'_x", "EN'_x", "M.P.'_x", "B.P.'_x", "H_FUS'_x", "DENSITY'_x",
       "IE'_x", "SE'_x", "AN'_y", "AM'_y", "G'_y", "P'_y", "R'_y", "EN'_y",
       "M.P.'_y", "B.P.'_y", "H_FUS'_y", "DENSITY'_y", "IE'_y", "SE'_y",
       "Pauling_x", "WorkFunction_x", "dBandCenter_x", "Pauling_y",
       "WorkFunction_y", "dBandCenter_y"]


# plt.bar(feature_name, final_importance, label = "Feature Importance")


# fig = plt.figure(figsize=(10,10))
# ax = fig.add_axes([0,0,1,1])
# feature_name  = ["AN'_x", "AM'_x", "G'_x", "P'_x",
#        "R'_x", "EN'_x", "M.P.'_x", "B.P.'_x", "H_FUS'_x", "DENSITY'_x",
#        "IE'_x", "SE'_x", "AN'_y", "AM'_y", "G'_y", "P'_y", "R'_y", "EN'_y",
#        "M.P.'_y", "B.P.'_y", "H_FUS'_y", "DENSITY'_y", "IE'_y", "SE'_y",
#        "Pauling_x", "WorkFunction_x", "dBandCenter_x", "Pauling_y",
#        "WorkFunction_y", "dBandCenter_y"]
# ax.bar(feature_name, final_importance)
# # plt.plot(feature_name, final_importance)
# plt.xlabel(feature_name, labelpad = 2)
# plt.show()


with open('GBR_C_111_results.txt', 'w') as f:
    print('Test_error:', str(sum(errors)/float(len(errors))), file=f)  # Python 3.x
    print('Train_error:', str(sum(errors_train)/float(len(errors_train))), file=f)  # Python 3.x
