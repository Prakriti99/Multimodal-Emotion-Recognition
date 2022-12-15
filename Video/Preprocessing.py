### General imports ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from time import sleep
import re
import os
import argparse
from collections import OrderedDict
import matplotlib.animation as animation

### Image processing ###
from scipy.ndimage import zoom
from scipy.spatial import distance
import imutils
from scipy import ndimage
import cv2
import dlib
from __future__ import division
from imutils import face_utils


import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, SeparableConv2D
from keras.utils import np_utils
from keras.regularizers import l2
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras import models
from keras.utils.vis_utils import plot_model
from keras.layers import Input, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras import layers


from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier


import h5py
from keras.models import model_from_json
import pickle

path = '/Users/tarun/Downloads/Project/Video'

pd.options.mode.chained_assignment = None  


dataset = pd.read_csv(local_path + 'fer2013.csv')


train = dataset[dataset["Usage"] == "Training"]


test = dataset[dataset["Usage"] == "PublicTest"]


train['pixels'] = train['pixels'].apply(lambda image_px : np.fromstring(image_px, sep = ' '))
test['pixels'] = test['pixels'].apply(lambda image_px : np.fromstring(image_px, sep = ' '))
dataset['pixels'] = dataset['pixels'].apply(lambda image_px : np.fromstring(image_px, sep = ' '))

X_train = train.iloc[:, 1].values
y_train = train.iloc[:, 0].values

X_test = test.iloc[:, 1].values
y_test = test.iloc[:, 0].values

X = dataset.iloc[:,1].values
y = dataset.iloc[:,0].values


X_train = np.vstack(X_train)
X_test = np.vstack(X_test)
X = np.vstack(X)


X_train = np.reshape(X_train, (X_train.shape[0],48,48,1))
y_train = np.reshape(y_train, (y_train.shape[0],1))

X_test = np.reshape(X_test, (X_test.shape[0],48,48,1))
y_test = np.reshape(y_test, (y_test.shape[0],1))

X = np.reshape(X, (X.shape[0],48,48,1))
y = np.reshape(y, (y.shape[0],1))

print("Shape of X_train and y_train is " + str(X_train.shape) +" and " + str(y_train.shape) +" respectively.")
print("Shape of X_test and y_test is " + str(X_test.shape) +" and " + str(y_test.shape) +" respectively.")


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X = X.astype('float32')

X_train /= 255
X_test /= 255
X /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y = to_categorical(y)

classes = np.unique(y_train)
nClasses = len(classes)
print('Total number of outputs : ', nClasses)
print('Output classes : ', classes)


nRows,nCols,nDims = X_train.shape[1:]
input_shape = (nRows, nCols, nDims)

def get_label(argument):
    labels = {0:'Angry', 1:'Disgust', 2:'Sadness', 3:'Joy', 4:'Neutral' , 5:'Surprise', 6:'Fear'}
    return(labels.get(argument, "Invalid emotion"))
    plt.figure(figsize=[10,5])

plt.subplot(121)
plt.imshow(np.squeeze(X_train[25,:,:], axis = 2), cmap='gray')
plt.title("Ground Truth : {}".format(get_label(int(y_train[0]))))

plt.subplot(122)
plt.imshow(np.squeeze(X_test[26,:,:], axis = 2), cmap='gray')
plt.title("Ground Truth : {}".format(get_label(int(y_test[1500]))))
np.save(local_path + 'X_train', X_train)
np.save(local_path + 'X_test', X_test)
np.save(local_path + 'X', X)
np.save(local_path + 'y_train', y_train)
np.save(local_path + 'y_test', y_test)
np.save(local_path + 'y', y)
X_train = np.load(local_path + "X_train.npy")
X_test = np.load(local_path + "X_test.npy")
y_train = np.load(local_path + "y_train.npy")
y_test = np.load(local_path + "y_test.npy")

shape_x = 48
shape_y = 48
nRows,nCols,nDims = X_train.shape[1:]
input_shape = (nRows, nCols, nDims)
classes = np.unique(y_train)
nClasses = len(classes)

model = OneVsRestClassifier(XGBClassifier())
model.fit(X_train.reshape(-1,48*48*1)[:5000], y_train[:5000])
gray = cv2.cvtColor(model.feature_importances_.reshape(shape_x, shape_y,3), cv2.COLOR_BGR2GRAY)
plt.figure(figsize=(12,8))
sns.heatmap(gray)
plt.show()