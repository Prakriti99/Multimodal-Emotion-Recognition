### General imports ###
import os
from glob import glob
import pickle
import itertools
import numpy as np
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

### Graph imports ###
import matplotlib.pyplot as plt
from PIL import Image

### Audio import ###
import librosa
import IPython
from IPython.display import Audio
import pickle

import pandas as pd

def frame(x, win_step=128, win_size=64):
    nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
    frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
    for t in range(nb_frames):
        frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
    return frames


def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    return mel_spect

def noisy_signal(signal, snr_low=15, snr_high=30, nb_augmented=2):
    
    # Signal length
    signal_len = len(signal)

    # Generate White noise
    noise = np.random.normal(size=(nb_augmented, signal_len))
    
    # Compute signal and noise power
    s_power = np.sum((signal / (2.0 ** 15)) ** 2) / signal_len
    n_power = np.sum((noise / (2.0 ** 15)) ** 2, axis=1) / signal_len
    
    # Random SNR: Uniform [15, 30]
    snr = np.random.randint(snr_low, snr_high)
    
    # Compute K coeff for each noise
    K = np.sqrt((s_power / n_power) * 10 ** (- snr / 10))
    K = np.ones((signal_len, nb_augmented)) * K
    
    # Generate noisy signal
    return signal + K.T * noise

filepath="/content/drive/MyDrive/CS5100/data/MELD.Raw/audio_emotion.pkl"
train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(filepath, 'rb'))
#print(train_audio_emb['0'].shape)
csv_path="/content/drive/MyDrive/CS5100/data/MELD.Raw/train_sent_emo.csv"
au_fp="/content/drive/MyDrive/CS5100/data/MELD.Raw/train_splits/audio/*.mp3"
au_files=glob(au_fp)
#print(au_files)
df=pd.read_csv(csv_path)
dictionary={"Audio":[],
     "Emotion":[],
     "Sentiment":[]
    }
df1=pd.DataFrame(dictionary)
#print(df1)
for index, row in df.iterrows():
    audiofile=[au_fp,"/dia",str(row['Dialogue_ID']),"_utt",str(row['Utterance_ID']),".mp3"]
    emo=row["Emotion"]
    sen=row["Sentiment"]
    op={"Audio":''.join(audiofile),"Emotion":emo,"Sentiment":sen}
    df1 = df1.append(op, ignore_index = True)

df_final=pd.read_csv("/content/drive/MyDrive/CS5100/data/MELD.Raw/train_final.csv")
# Audio file path and names
file_path = "/content/drive/MyDrive/CS5100/data/MELD.Raw/train_splits/audio"
file_names = df_final["Audio"].tolist()
emotions=df_final["Emotion"].tolist()
print(df_final)
# Initialize features and labels list
signal = []
labels = []
final_fn=[]
# Sample rate (16.0 kHz)
sample_rate = 16000     

# Max pad lenght (3.0 sec 49100) V1=5 sec, 81000
max_pad_len = 81000

for audio_index, audio_file in enumerate(file_names):
    try:

        print(audio_file)
        y, sr = librosa.core.load(audio_file, sr=sample_rate, offset=0.5)

        # Z-normalization
        y = zscore(y)

        # Padding or truncated signal 
        if len(y) < max_pad_len:    
            y_padded = np.zeros(max_pad_len)
            y_padded[:len(y)] = y
            y = y_padded
        elif len(y) > max_pad_len:
            y = np.asarray(y[:max_pad_len])

        # Add to signal list
        signal.append(y)

        # Set label
        labels.append(emotions[audio_index])
        final_fn.append(audio_file)
        print(emotions[audio_index])
        # Print running...
    except Exception as e:
        print(e)
        pass
        
labels = np.asarray(labels).ravel()

#Augment data
augmented_signal = list(map(noisy_signal, signal))

mel_spect = np.asarray(list(map(mel_spectrogram, signal)))
augmented_mel_spect = [np.asarray(list(map(mel_spectrogram, augmented_signal[i]))) for i in range(len(augmented_signal))]
##Building dataset
MEL_SPECT_train, MEL_SPECT_test, AUG_MEL_SPECT_train, AUG_MEL_SPECT_test, label_train, label_test = train_test_split(mel_spect, augmented_mel_spect, labels, test_size=0.2)

# Build augmented labels and train
aug_label_train = np.asarray(list(itertools.chain.from_iterable([[label] * nb_augmented for label in label_train])))
AUG_MEL_SPECT_train = np.asarray(list(itertools.chain.from_iterable(AUG_MEL_SPECT_train)))

# Concatenate original and augmented
X_train = np.concatenate((MEL_SPECT_train, AUG_MEL_SPECT_train))
y_train = np.concatenate((label_train, aug_label_train))

# Build test set
X_test = MEL_SPECT_test
y_test = label_test


win_ts = 128
hop_ts = 64


X_train = frame(X_train, hop_ts, win_ts)
X_test = frame(X_test, hop_ts, win_ts)

pickle.dump(X_train.astype(np.float16), open('trainX.p', 'wb'))
pickle.dump(y_train, open('trainY.p', 'wb'))
pickle.dump(X_test.astype(np.float16), open('testX.p', 'wb'))
pickle.dump(y_test, open('testY.p', 'wb'))