'''
Audio processing script (Type-2)

This is MFCC dump2

This part is slow and tedious (genrally), thus we wish to perform it only once.
At the end of this script we store the data and corresponding labels as two
seperate pickle dump format .npy files.

When doing the machine learning part we load them and use them, thus saving time.

'''

# importing dependencies
import pandas as pd # data frame
import numpy as np # matrix math
from scipy.io import wavfile # reading the wavfile
import os # interation with the OS
from sklearn.utils import shuffle # shuffling of data
from random import sample # random selection
from tqdm import tqdm # progress bar

#audio processing
from scipy import signal # audio processing
from scipy.fftpack import dct

PATH = '/home/bonde_yash97/speech/train/audio/'
PATH_SAVE_DATA_1 = '/home/bonde_yash97/speech/pickles/mfcc_m1.npy'
PATH_SAVE_DATA_2 = '/home/bonde_yash97/speech/pickles/mfcc_m2.npy'
PATH_SAVE_OH = '/home/bonde_yash97/speech/pickles/mfcc_oh.npy'
PATH_SAVE_TEST = '/home/bonde_yash97/speech/pickles/x.npy'

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

'''
Functions to obtain features.
This has been learned from Haythem Fayek's amazing blog post explaining all the features in detail
Link: http://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
'''
def load_signal(path_file):
	sample_rate, signal = wavfile.read(path_file)
	pre_emphasis = 0.97
	emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
	return emphasized_signal, sample_rate

def framing(emphasized_signal, sample_rate):
	# some paramters
	frame_size = 0.05
	frame_stride = 0.03

	frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
	signal_length = len(emphasized_signal)
	frame_length = int(round(frame_length))
	frame_step = int(round(frame_step))
	num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

	pad_signal_length = num_frames * frame_step + frame_length
	z = np.zeros((pad_signal_length - signal_length))
	pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

	indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
	frames = pad_signal[indices.astype(np.int32, copy=False)]

	# applying hamming window
	frames *= np.hamming(frame_length)
	return frames

def filter_banks_func(frames, sample_rate):
	NFFT = 512
	mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
	pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

	nfilt = 40

	low_freq_mel = 0
	high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
	mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
	hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
	bin = np.floor((NFFT + 1) * hz_points / sample_rate)

	fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
	for m in range(1, nfilt + 1):
		f_m_minus = int(bin[m - 1])   # left
		f_m = int(bin[m])             # center
		f_m_plus = int(bin[m + 1])    # right

		for k in range(f_m_minus, f_m):
			fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
		for k in range(f_m, f_m_plus):
			fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
	filter_banks = np.dot(pow_frames, fbank.T)
	filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
	filter_banks = 20 * np.log10(filter_banks)  # dB

	return filter_banks

def normalize_filter_banks(fb):
	fb -= (np.mean(fb, axis=0) + 1e-8)
	return fb

def MFCC(filter_banks):
	num_ceps = 20
	cep_lifter = 22 # general value

	mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13
	(nframes, ncoeff) = mfcc.shape
	n = np.arange(ncoeff)
	lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
	mfcc *= lift 

	# mean normalisation
	mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
	return mfcc

def get_features(path):
	signal, sample_rate = load_signal(path)
	frames = framing(signal, sample_rate)
	f_banks = filter_banks_func(frames, sample_rate)
	return MFCC(f_banks), normalize_filter_banks(f_banks)

'''
Rest of the code
'''

def paths_to_data(paths, word2id, unk = False):
	data_1 = np.zeros(shape = (len(paths), 32, 20))
	data_2 = np.zeros(shape = (len(paths), 32, 40))
	labels = []
	indexes = []
	for i in tqdm(range(len(paths))):
		f = paths[i]
		audio, fb = get_features(paths[i])
		if audio.shape != (32,20) or fb.shape != (32,40):
			indexes.append(i)
		elif audio.shape == (32,20) and fb.shape == (32,40):
			data_1[i] = audio
			data_2[i] = fb
		# mode, if unk is set we are doing it for unknown files
			if unk == True:
				labels.append(word2id['unknown'])
			else:
				labels.append(word2id[f.split('/')[-2]])

	print('Number of instances with inconsistent shape:', len(indexes))

	return data_1[:len(paths)-len(indexes)], data_2[:len(paths)-len(indexes)], labels, indexes

# for labled data
files = train.loc[train['label'] != 'unknown']['file'].values
print("[!]For labled data...")
data_1, data_2, labels, i = paths_to_data(files, word2id)

# for unknown files
print("[!]For labled data (unk)...")
unk_data_1, unk_data_2, unk_labels, i = paths_to_data(unk_files, word2id, unk = True)

# merging the two data sources
data_1 = np.vstack([data_1, unk_data_1])
data_2 = np.vstack([data_2, unk_data_2])
labels = np.hstack([labels, unk_labels])

# reshaping the data for CNN
# for MFCC
s = data_1.shape
data_1 = np.reshape(data_1, [s[0], s[1], s[2], 1])
# for filter banks
s = data_2.shape
data_2 = np.reshape(data_2, [s[0], s[1], s[2], 1])

print("[*] data_1.shape:", data_1.shape)
print("[*] data_2.shape:", data_2.shape)
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
data_1, data_2, one_hot_labels = shuffle(data_1, data_2, one_hot_labels)

# saving the np arrays
print('[!]Saving data_1 file at:', PATH_SAVE_DATA_1, ' ...')
np.save(PATH_SAVE_DATA_1 , data_1)

print('[!]Saving data_2 file at:', PATH_SAVE_DATA_2, ' ...')
np.save(PATH_SAVE_DATA_2 , data_2)

print('[!]Saving one-hot-label file at:', PATH_SAVE_OH, ' ...')
np.save(PATH_SAVE_OH , one_hot_labels)

print('...Files are saved')