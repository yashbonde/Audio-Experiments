'''
test spec data dump file
'''
import numpy as np # matrix math
from scipy import signal # audio processing
from scipy.io import wavfile # reading the wavfile
from glob import glob # file management
from tqdm import tqdm # progress bar

PATH_TEST = '../speech/test/audio/'
PATH_MODEL = '../speech/final_1_04.h5'
PATH_TEST_SAVE = '../speech/pickles/test_data.npy'

# files
files = glob(PATH_TEST + '*.wav')
print('[*]Number of files:', len(files))

# feature extraction
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

def paths_to_data(paths):
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

	print('Number of instances with inconsistent shape:', len(indexes))

	return data

# getting data
data = paths_to_data(files)

# saving the data
print('[!]Saving data file at:', PATH_TEST_SAVE, ' ...')
np.save(PATH_TEST_SAVE)
print('... Files saved!')
