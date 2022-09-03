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
from classes import preprocess_text
from functions import get_dataframe

file_train = os.path.join(dirname,'complete_dataset.csv')
corpus = get_dataframe(file_train)
# corpus = corpus.iloc[0:500]
X_unsuper = corpus['Paragraph'].astype(str)
print("Preprocessing unsupervised data ........")
X_unsuper = preprocess_text(X_unsuper)
np.save(os.path.join(dirname, 'preprocess_text'), X_unsuper)
print("Preproces features have been saved Successfully !!!")
