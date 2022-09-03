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

# url = "/Users/shivamsaxena/Desktop/MTP/ML/single_point_alloy/Cu/Cu_O.csv"
# url = 'Cu_O (1).csv'
url = 'featured_data_111_C.csv'
df = pd.read_csv(url, header=0)
df = df.drop(['Unnamed: 0'],axis=1)
# print(df.head(5))
# Y=df.loc[:,'B.E.']
# X= df.loc[:,'AN':'SE']

Y = df.loc[:,'reactionEnergy']
X = df.loc[:, 'AN' : 'dBandCenter']

errors=[]
errors_train=[]
feature_importances_array=np.zeros(28)
i=0
error_list=[]
error_all_i=np.zeros(28)
for i in range(0,100):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    


#    regr = GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3)
#    
#    clf=regr.fit(X_train, y_train)
#    importances = clf.feature_importances_
    
#    print importances

#    X_train = pca.transform(X_train)
#    X_test = pca.transform(X_test)
    
    parameter_candidates = [
            {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth':[2]},

                ]
    n_folds = 5
    clf = GridSearchCV(GradientBoostingRegressor(),cv=n_folds, param_grid=parameter_candidates)
    model=clf.fit(X_train, y_train)
#    print('Best :',model.best_estimator_.n_estimators,model.best_estimator_.learning_rate,model.best_estimator_.max_depth)
    y_train_predicted=model.predict(X_train)
    y_test_predicted=model.predict(X_test)

    # Later find which values create most errors
    errors.append(sqrt(mean_squared_error(y_test, y_test_predicted)))
    errors_train.append(sqrt(mean_squared_error(y_train, y_train_predicted)))
    b=model.best_estimator_.feature_importances_
    feature_importances_array=np.add(feature_importances_array,b)
    i=i+1
    print(i)

    
#print "2"    
print ("Train Error" + str(sum(errors_train)/float(len(errors_train)))) 
print ("Test Error" + str(sum(errors)/float(len(errors))))
final_importance=np.divide(feature_importances_array,100.0)