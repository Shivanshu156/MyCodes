import os
import sys
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


dirname = os.path.dirname(__file__)
# dirname = r'C:\Users\sverma9\Desktop\Shivanshu\.git\Contract-Analysis'
sys.path.insert(0, dirname)
from functions import get_dataframe, cm_analysis
from functions import NLTKPreprocessor, Best_estimator_grid

file_train = os.path.join(dirname,'complete_dataset.csv')

n_clusters = int(sys.argv[1])
kmeans_model_name = 'KMeans_Clusters_'+str(n_clusters) + '.pickle'

X_unsuper = np.load(os.path.join(dirname, 'preprocess_text.npy'))
print("preprocess_text features loaded successfully")
clusters = KMeans(n_clusters=n_clusters,init='k-means++',n_init=10,n_jobs=2, random_state=None)
clusters.fit(X_unsuper)
dump(clusters, os.path.join(dirname, kmeans_model_name))
print("Model has been dumped Successfully !!!")
