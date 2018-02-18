'''
This file contains the script to make and train the model on initial data
'''

import numpy as np # matrix math
import librosa # audio processing
from glob import glob # file handling
import copy
import random

# Our default machine learning library will be keras
# using tensoflow backend here,
# you can use Theano or Microsoft CNTK here
from keras.models import Sequential 
from keras.layers import Conv2D, Conv1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten, Reshape

# Step 0: Define the constants
SAMPLE_RATE = 8000
FRAME_TIME = 100 # mili-seconds

# Step 1: Import the data
print("[!]Importing data...")
# Since ring is a different file we are loading differently
ring_file = '~/Train_Synthetic/16_only_rings.wav'

# rest of the hold files
files_hold = glob('~/moretrainingsamplesforholdmusic/*.*')
del files_hold[5] # ivrs one
files_hold.append('~/Main_Hold_Music/musiconhold_popular.wav')
hold_audio = []
hold_audio_sr = []
for f in files_hold:
	a, s = librosa.load(f)
	hold_audio.append(a)
	hold_audio_sr.append(s)

# Step 1.5: Process the files
print("[!]Pre-processing the files...")
# This includes:
# 1) resampling the files to consistent sample_rate for all the audio
# 2) converting the audio from stereo to mono
ring_audio, ring_sr = librosa.load(ring_file)
ring_audio = librosa.core.to_mono(ring_audio)
ring_audio = librosa.core.resample(ring_audio, ring_sr, SAMPLE_RATE)

for i in range(len(hold_audio)):
	hold_audio[i] = librosa.core.to_mono(hold_audio[i])
	hold_audio[i] = librosa.core.resample(hold_audio[i], hold_audio_sr[i], SAMPLE_RATE)

# Step 2: Segment the data
print("[!]Segmenting the data...")
def break_to_segments(y):
	num_segments = int((len(y) * 1000) / (SAMPLE_RATE * FRAME_TIME))
	segsamples = int(SAMPLE_RATE * FRAME_TIME / 1000)
	segments = []
	for i in range(num_segments):
		seg = y[i*segsamples: (i+1)*segsamples]
		segments.append(seg)
	return segments

def crude_ad(x):
	'''
	Crude activity detection function
	'''
	def energy_threshold_detector(x, threshold = -35):
	    '''
	    return True if the median value in the array is above a certain threshold
	    '''
	    log_val = 10 * np.log10(x ** 2)
	    if np.max(log_val) >= threshold:
	        return True
	    return False

	# performing the crude_ad
	segments_ad = []
	segments_broken = break_to_segments(ring_audio)
	for i,seg in enumerate(segments_broken):
	    segments_ad.append(energy_threshold_detector(seg))

	seg_true = [i for i,act in enumerate(segments_ad) if act == True]

	return [segments_broken[i] for i in seg_true], [segments_broken[i] for i in range(len(segments_broken)) if i not in seg_true]

# first we segment useful samples from 
ring_samples, silence_samples = crude_ad(ring_audio)

hold_samples = []
for ha in hold_audio:
	hold_samples.extend(break_to_segments(ha))

# Step 3: Complie final data
print("[!]Compiling final data...")
ring_labels = [[1, 0, 0] for _ in range(len(ring_samples))]
hold_labels = [[0, 1, 0] for _ in range(len(hold_samples))]
sile_labels = [[0, 0, 1] for _ in range(len(silence_samples))]

data_ = copy.deepcopy(hold_samples)
data_.extend(ring_samples)
data_.extend(silence_samples)

labels_ = copy.deepcopy(hold_labels)
labels_.extend(ring_labels)
labels_.extend(sile_labels)

# Step 4: Convert data to audio features
print("[!]Converting to audio features...")
def convert_to_features(x):
	'''
	For each single array x return the log spectrogram
	'''
	return librosa.feature.melspectrogram(x, SAMPLE_RATE)

for i in range(len(data_)):
	data_[i] = convert_to_features(data_[i])

# reshaping the data
data_ = np.array(data_)
data_ = np.reshape(data_, [-1, 128, 2, 1])
labels_ = np.array(labels_)

# Step 5: Design the neural network model
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (2, 2), activation = 'relu', input_shape = (128, 2, 1)))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(3, activation = 'softmax'))
# model.summary()
model.compile(loss = 'mean_squared_error', optimizer = 'RMSprop', metrics = ['accuracy'])

# Step 6: Train the model on example data
print("[!]Training the model...")
# shuffle the data
random.shuffle([data_, labels_])

model.fit(x = data_, y = labels_, epochs = 5, validation_split = 0.2, batch_size = 128)

# Step 7: Save the model
path_save = '/Users/yashj.bonde/Desktop/ML/Audio_Vivek/final/Models/'
file_name = 'model_1.h5'

model.save(path_save + file_name)
