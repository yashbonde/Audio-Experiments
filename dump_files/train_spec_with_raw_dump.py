'''
Audio processing script (Type-2)

This is spectrogram dump

This part is slow and tedious (genrally), thus we wish to perform it only once.
At the end of this script we store the data and corresponding labels as two
seperate pickle dump format .npy files.

When doing the machine learning part we load them and use them, thus saving time.

'''

# importing dependencies
import pandas as pd # data frame
import numpy as np # matrix math
from scipy import signal # audio processing
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar
from glob import glob # file handling

PATH = '../speech/pickles/train/audio/'
PATH_SIL = '../speech/train/audio/_background_noise_/'
PATH_SAVE_DATA = '../speech/pickles/spec_data.npy'
PATH_SAVE_OH = '../speech/pickles/spec_one_hot.npy'

def load_files_wo_back(path):
	# write the complete file loading function here, this will return
	# a dataframe having files and labels
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
		return path + label + '/' + fname[len(label)+1:]

	train['file'] = train.apply(lambda x: remove_label_from_file(*x), axis=1)
	train['label'] = train['folder'].apply(lambda x: x if x in labels_to_keep else 'unknown')

	labels_to_keep.append('unknown')

	return train, labels_to_keep

train, labels_to_keep = load_files_wo_back(PATH)

# making word2id dictr
word2id = dict((c,i+1) for i,c in enumerate(sorted(labels_to_keep)))

print(word2id)

# get some files which will be labeled as unknown
unk_files = train.loc[train['label'] == 'unknown']['file'].values
# randomly selecting 3000 files
unk_files = sample(list(unk_files), 3000)

# loading data with silence
def load_files_w_back():
	sil_files = glob(PATH_SIL + '*.wav')
	# silence background sample
	all_sil = []
	for s in files:
		sr, audio = wavfile.read(s)
		# converting the file into samples of 1 sec each
		len_ = int(len(audio)/sr)
		for i in range(len_-1):
			sample_ = audio[i*sr:(i+1)*sr]
			all_sil.append(sample_)
	sil_data =  np.zeros((392, 16000, ))
	for i,d in enumerate(all_sil):
		sil_data[i] = d
	return sil_data

# the feature extraction
def log_specgram(audio, sample_rate, window_size=10, 
	step_size=10, eps=1e-10):
	nperseg = int(round(window_size * sample_rate / 1e3))
	noverlap = int(round(step_size * sample_rate / 1e3))
	_, _, spec = signal.spectrogram(audio, fs=sample_rate,
		window='hann', nperseg=nperseg, noverlap=noverlap,
		detrend=False)
	return np.log(spec.T.astype(np.float32) + eps)

def audio_to_data(path):
	# we take a single path and convert it into data
	sample_rate, audio = wavfile.read(path)
	spectrogram = log_specgram(audio, sample_rate, 10, 0)
	return spectrogram.T

def paths_to_data(paths, word2id, unk = False):
	data = np.zeros(shape = (len(paths), 81, 100))
	labels = []
	indexes = []
	for i in tqdm(range(len(paths))):
		f = paths[i]
		audio = audio_to_data(paths[i])
		if audio.shape != (81,100):
			indexes.append(i)
			# now we need to pad the sequence such that we have consistent data
			len_diff = 81 - audio.shape[0]
			if len_diff > 0:
				pad = np.zeros((len_diff, 100))
				audio = np.vstack([audio, pad])
			elif len_diff < 0:
				audio = audio[:81]

			width_diff = 100 - audio.shape[1]
			if width_diff > 0:
				pad = np.zeros((81, width_diff))
				audio = np.hstack([audio, pad])
			elif width_diff < 0:
				audio = audio[:,:100]

		# now adding it to the data
		data[i] = audio

		# unk stands for unknown files
		if unk == True:
			labels.append(word2id['unknown'])
		else:
			labels.append(word2id[f.split('/')[-2]])

	print('Number of instances with inconsistent shape:', len(indexes))

	return data, labels, indexes

# for labled data
files = train.loc[train['label'] != 'unknown']['file'].values
print("[!]For labled data...")
data, labels, i = paths_to_data(files, word2id)

# for unknown files
print("[!]For labled data (unk)...")
unk_data, unk_labels, i = paths_to_data(unk_files, word2id, unk = True)

# for silence
print("[!]For labled data (sil)...")
sil_data = load_files_w_back()
silence = np.zeros((392, 81, 100))
for i,s in enumerate(sil_data):
	silence[i] = log_specgram(s, 16000, 10, 0).T
silence_labels = np.hstack([word2id['silence']] * silence.shape[0])

# merging the two data sources
data = np.vstack([data, unk_data, silence])
labels = np.hstack([labels, unk_labels, silence_labels])

'''# reshaping the data for CNN
s = data.shape
data = np.reshape(data, [s[0], s[1], s[2]])'''

print("[*] data.shape:", data.shape)
print("[*] labels.shape:", labels.shape)

# covert labels to one-hot encoding
def make_onehot(seq):
	one_hot = np.zeros(shape = (len(seq), max(seq)))
	for i,s in enumerate(seq):
		one_hot[i][s-1] = 1.
	return one_hot

one_hot_labels = make_onehot(labels)

print("[*] one_hot_labels.shape:", one_hot_labels.shape)

# shuffling files
data, one_hot_labels = shuffle(data, one_hot_labels)

# saving the numpy arrays
print('[!]Saving data file at:', PATH_SAVE_DATA, ' ...')
np.save(PATH_SAVE_DATA , data)

print('[!]Saving one-hot-label file at:', PATH_SAVE_OH, ' ...')
np.save(PATH_SAVE_OH , one_hot_labels)

print('...Files are saved')
