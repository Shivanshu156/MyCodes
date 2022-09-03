import os
import sys
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.mixture import BayesianGaussianMixture


dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis
from functions import NLTKPreprocessor, Best_estimator_grid

file_train = os.path.join(dirname,'complete_dataset.csv')

n_components = int(sys.argv[1])
gmm_model_name = 'GMM_Components_'+str(n_components) + '.pickle'

X_unsuper = np.load(os.path.join(dirname, 'preprocess_text.npy'))
print("preprocess_text features loaded successfully")
GMM_model = BayesianGaussianMixture(n_components= n_components,verbose=10, random_state=None)
GMM_model.fit(X_unsuper)
dump(GMM_model, os.path.join(dirname, gmm_model_name))
print("Model has been dumped Successfully !!!")
