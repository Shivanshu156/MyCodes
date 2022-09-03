import os, sys
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.svm import SVR
from joblib import load, dump
import matplotlib.pyplot as plt
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


dirname = os.path.dirname(__file__)
# dirname = '/Users/shivanshu/Desktop/Sem IX/MTP'
sys.path.insert(0, dirname)
filename = os.path.join(dirname, 'O_111_featured_data.csv')
GBR_model_name = os.path.join(dirname, 'GBR_O_111.pickle')


data = pd.read_csv(filename)
data = data.drop(['Unnamed: 0'],axis=1)
data.dropna(inplace=True)
data  = data.loc[data['B.E.']<0]
Y = data.loc[:,'B.E.'].values
X = data.loc[:, "AN" : "SE'"].values
X = data.loc[:, ['G','EN','M.P.','B.P.','H_FUS','IE', "SE","G'","EN'","M.P.'","B.P.'","H_FUS'","IE'", "SE'","ElectronAffinity", "Pauling", "WorkFunction", "dBandCenter" ]]
X = data.loc[:, ["SE", "IE","SE'", "IE'"] ]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, shuffle=True)


best_model = load(os.path.join(dirname, GBR_model_name))
best_model.fit(X_train,y_train)
y_train_predicted = best_model.predict(X_train)
y_test_predicted = best_model.predict(X_test)

def percent_error(y_true, y_pred):
       error  = (y_true - y_pred)/y_true
       error = np.square(error)
       error = np.sqrt(error)
       outliners = 0
       for i in range(0, len(error)):
              if abs(error[i]) > 1:
                     # print('here')
                     outliners = outliners+1
                     error[i] = 0
       ans = np.sum(error)
       ans = ans/len(error)
       return ans, outliners

percent_train, outliner_train = percent_error(y_train, y_train_predicted)
percent_test, outliner_test = percent_error(y_test, y_test_predicted)


errors=[]
errors_train=[]
feature_importances_array=np.zeros(24)
error_list=[]
error_all_i=np.zeros(24)
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
    print(i)

final_importance=np.divide(feature_importances_array,100.0)

print ("Percentage Train Error", percent_train*100, "% with", outliner_train, "of",len(y_train), "outliner points") 
print ("Percentage Test Error", percent_test*100, "% with", outliner_test,"of",len(y_test), "outliner points")
print('RMSE Test_error:', sum(errors)/float(len(errors)), "eV")  # Python 3.x
print('RMSE Train_error:', sum(errors_train)/float(len(errors_train)), "eV")  # Python 3.x
print('R square score for test data is ',best_model.score(X_test, y_test))
print('R square score for train data is ',best_model.score(X_train, y_train))

features = ['Atomic no','Atomic mass','Group', 'Period', 'Radius', 'Electronegativity', 'Melting Point', 'Boiling Point','Fusion Heat',
             'Density', 'Ionisation Energy', 'Surface Energy','Atomic no','Atomic mass','Group', 'Period', 'Radius', 'Electronegativity', 'Melting Point', 'Boiling Point','Fusion Heat',
             'Density', 'Ionisation Energy', 'Surface Energy' ]
color = ['k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r']
print(final_importance)

y_pos = np.arange(len(features))
plt.bar(y_pos, final_importance)
plt.xticks(y_pos, features)
plt.show()

