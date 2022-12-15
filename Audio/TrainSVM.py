### General imports ###
from glob import glob
import os
import pickle
import itertools
import pandas as pd
import numpy as np

### Warning import ###
import warnings
warnings.filterwarnings('ignore')

### Graph imports ###
import matplotlib.pyplot as plt

### Sklearn imports ###
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


[features, labels] = pickle.load(open("/content/drive/MyDrive/CS5100/Audio train test pickle files/trainSVM.p", "rb"))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=123)

# Encode Label from categorical to numerical
lb = LabelEncoder()
lb.fit(y_train)
y_train, y_test = lb.transform(y_train), lb.transform(y_test)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Kbest = SelectKBest(k="all")
selected_features = Kbest.fit(X_train, y_train)

alpha=0.01
X_train = X_train[:,np.where(selected_features.pvalues_ < alpha)[0]]
X_test = X_test[:,np.where(selected_features.pvalues_ < alpha)[0]]

pca = PCA(n_components=130)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

G_list = [0.001, 0.005, 0.01]
C_list = [1, 2, 3, 4, 5, 7, 10, 20, 50]

parameters = [{'kernel': ['rbf'], 'C': C_list, 'gamma': G_list}]

model = SVC(decision_function_shape='ovr')

# Cross Validation 
cv = GridSearchCV(model, parameters, cv=3, verbose=0, n_jobs=-1).fit(X_train, y_train)
model = SVC(kernel='rbf', C=3, gamma=0.005, decision_function_shape='ovr').fit(X_train, y_train)
pred = model.predict(X_test)
score = model.score(X_test, y_test)

pred = (lb.inverse_transform((pred.astype(int).flatten())))
actual = (lb.inverse_transform((y_test.astype(int).flatten())))

df_pred = pd.DataFrame({'Actual': actual, 'Prediction': pred})

print('Accuracy Score on test dataset: {}%'.format(np.round(100 * score,2)))
pickle.dump(model, open('/content/drive/MyDrive/CS5100/MODEL_CLASSIFIER.p', 'wb'))