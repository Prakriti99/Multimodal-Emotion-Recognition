from glob import glob
import os
import pickle
import itertools
import numpy as np

### Audio preprocessing imports ###
import sys
sys.path.insert(0,'/content/drive/MyDrive/CS5100/Multimodal-Emotion-Recognition-master/01-Audio/Notebook/SVM')
from AudioLibrary.AudioSignal import *
from AudioLibrary.AudioFeatures import *
import pandas as pd

label_dict = {'01': 'anger', '02':'disgust', '03':'fear', '04':'joy', '05':'neutral', '06':'sadness', '07':'surprise'}


def global_feature_statistics(y, win_size=0.025, win_step=0.01, nb_mfcc=12, mel_filter=40,
                             stats = ['mean', 'std', 'med', 'kurt', 'skew', 'q1', 'q99', 'min', 'max', 'range'],
                             features_list =  ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', 'spectral_entropy', 'spectral_flux', 'sprectral_rolloff', 'mfcc']):
    
    # Extract features
    audio_features = AudioFeatures(y, win_size, win_step)
    features, features_names = audio_features.global_feature_extraction(stats=stats, features_list=features_list)
    return features


df_final=pd.read_csv('/content/drive/MyDrive/CS5100/data/MELD.Raw/train_final.csv')
print(df_final)
# Audio file path and names
#file_path = r"C:\Users\khush\OneDrive\Desktop\Prakriti_CS5100\MELD.Raw.tar\MELD.Raw\train_splits\audio"
file_names = df_final["Audio"].tolist()
emotions=df_final["Emotion"].tolist()
# Initialize signal and labels list
signal = []
labels = []

# Sample rate (44.1 kHz)
sample_rate = 35000
# Compute global statistics features for all audio file
for audio_index, audio_file in enumerate(file_names):
      
    # Read audio file
    signal.append(AudioSignal(sample_rate, filename=audio_file))
    
    # Set label
    labels.append(emotions[audio_index])

    # Print running...
    if (audio_index % 100 == 0):
        print("Import Data: RUNNING ... {} files".format(audio_index))
        
# Cast labels to array
labels = np.asarray(labels).ravel()
    
# Features extraction parameters
sample_rate = 16000 # Sample rate (16.0 kHz)
win_size = 0.025    # Short term window size (25 msec)
win_step = 0.01     # Short term window step (10 msec)
nb_mfcc = 12        # Number of MFCCs coefficients (12)
nb_filter = 40      # Number of filter banks (40)
stats = ['mean', 'std', 'med', 'kurt', 'skew', 'q1', 'q99', 'min', 'max', 'range'] # Global statistics
features_list =  ['zcr', 'energy', 'energy_entropy', 'spectral_centroid', 'spectral_spread', # Audio features
                      'spectral_entropy', 'spectral_flux', 'sprectral_rolloff', 'mfcc']
features = np.asarray(list(map(global_feature_statistics, signal)))

pickle.dump([features, labels], open("trainSVM.p", 'wb'))