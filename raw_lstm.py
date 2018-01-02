PATH = '/home/bonde_yash97/speech/train/audio'

import os
from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
# Audio manipulation
from scipy.io import wavfile # for loading the files
from scipy import signal # for getting spectrogram
# shuffling of data
from sklearn.utils import shuffle
# machine learning
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# loading the files
train_labels = os.listdir(PATH)
train_labels.remove('_background_noise_')

labels_to_keep = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'silence']

train_file_labels = dict()
for label in train_labels:
	files = os.listdir(PATH + '/' + label)
	for f in files:
		train_file_labels[label + '/' + f] = label

train = pd.DataFrame.from_dict(train_file_labels, orient='index')
train = train.reset_index(drop=False)
train = train.rename(columns={'index': 'file', 0: 'folder'})
train = train[['folder', 'file']]
train = train.sort_values('file')
train = train.reset_index(drop=True)

def remove_label_from_file(label, fname):
	return fname[len(label)+1:]

train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')

labels_to_keep.append('unknown')

# loading the audio
def log_specgram(audio, sample_rate, window_size=10, step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	_, _, spec = signal.spectrogram(audio, fs=sample_rate, window='hann',
		nperseg=nperseg, noverlap=noverlap, detrend=False)
	return np.log(spec.T.astype(np.float32) + eps)

# make labels and convert them into one hot encodings
labels = sorted(labels_to_keep)
word2id = dict((c,i) for i,c in enumerate(labels))
label = train['label'].values
label = [word2id[l] for l in label]

def make_one_hot(seq, n):
	# n --> vocab size
	seq_new = np.zeros(shape = (len(seq), n))
	for i,s in enumerate(seq):
		seq_new[i][s] = 1.
	return seq_new

one_hot_l = make_one_hot(label, 12)

# getting all the paths to the files
paths = []
folders = train['folder']
files = train['file']

for i in range(len(files)):
	path = str(PATH + '/' + str(folders[i]) + '/' + str(files[i]))
	paths.append(path)


def audio_to_data(path):
	# we take a single path and convert it into data
	sample_rate, audio = wavfile.read(path)
	spectrogram = log_specgram(audio, sample_rate, 10, 0)
	return spectrogram.T

def paths_to_data(paths,labels):
	data = np.zeros(shape = (len(paths), 81, 100))
	indexes = []
	print(type(paths[0]))
	for i in tqdm(range(len(paths))):
		audio = audio_to_data(paths[i])
		if audio.shape != (81,100):
			indexes.append(i)
		else:
			data[i] = audio
	final_labels = [l for i,l in enumerate(labels) if i not in indexes]
	print('Number of instances with inconsistent shape:', len(indexes))
	return data[:len(data)-len(indexes)], final_labels, indexes

d,l,indexes = paths_to_data(paths,one_hot_l)

labels = np.zeros(shape = [d.shape[0], len(l[0])])
for i,array in enumerate(l):
	for j, element in enumerate(array):
		labels[i][j] = element
print(labels.shape)

d,labels = shuffle(d,labels)

# Model
model = Sequential()
model.add(LSTM(256, input_shape = (81, 100)))
# model.add(Dense(1028))
model.add(Dropout(0.2))
model.add(Dense(128))
model.add(Dropout(0.2))
model.add(Dense(12, activation = 'softmax'))
model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])

# fitting the model
model.fit(x = d, y = labels, epochs = 5, batch_size = 128)

# saving the model for later use
model.save('model_1.h5')