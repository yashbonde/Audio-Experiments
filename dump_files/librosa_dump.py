'''
Audio processing script

This part is slow and tedious (>60mins), thus we wish to perform it only once.
At the end of this script we store the data and corresponding labels as two
seperate pickle dump format .npy files.

When doing teh machine learning part we load them and use them, thus saving time.

'''

# importing dependencies
import pandas as pd # data frame
import numpy as np # matrix math
import librosa # audio processing
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar

PATH = '/home/__/speech/train/audio/'
PATH_SAVE_DATA = '/home/__/speech/pickles/data.npy'
PATH_SAVE_OH = '/home/__/speech/pickles/one_hot.npy'
PATH_SAVE_TEST = '/home/__/speech/pickles/x.npy'

def test(path):
	x = np.random.random([124, 123, 122])
	np.save(path, x)
test(PATH_SAVE_TEST)

def load_files(path):
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

train, labels_to_keep = load_files(PATH)

# making word2id dictr
word2id = dict((c,i+1) for i,c in enumerate(sorted(labels_to_keep)))

print(word2id)

# get some files which will be labeled as unknown
unk_files = train.loc[train['label'] == 'unknown']['file'].values
# randomly selecting 3000 files
unk_files = sample(list(unk_files), 3000)

# Writing functions to extract the data, script from kdnuggets: 
# www.kdnuggets.com/2016/09/urban-sound-classification-neural-networks-tensorflow.html
def extract_feature(path):
	X, sample_rate = librosa.load(path)
	stft = np.abs(librosa.stft(X))
	mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
	chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
	mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
	contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
	tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
	return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(files, word2id, unk = False):
	# n: number of classes
	features = np.empty((0,193))
	labels = []
	fail_files = []
	for i in tqdm(range(len(files))):
		f = files[i]
		# some of the files are empty
		try:
			mfccs, chroma, mel, contrast,tonnetz = extract_feature(f)
			ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
			features = np.vstack([features,ext_features])
		except:
			fail_files.append(f)

		# mode, if unk is set we are doing it for unknown files
		if unk == True:
			labels.append(word2id['unknown'])
		else:
			labels.append(word2id[f.split('/')[-2]])
	return np.array(features), labels

# for labled data
files = train.loc[train['label'] != 'unknown']['file'].values
print("[!]For labled data...")
data, labels = parse_audio_files(files, word2id)

# for unknown files
print("[!]For labled data (unk)...")
unk_data, unk_labels = parse_audio_files(unk_files, word2id, unk = True)

# merging the two data sources
data = np.vstack([data, unk_data])
labels = np.hstack([labels, unk_labels])

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
