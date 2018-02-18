'''
This file contains the segmentation class
'''

# importing the dependencies
import numpy as np # matrix math
import os # system interaction
import librosa # audio processing file
import matplotlib.pyplot as plt # plotting of functions
from keras import models # loading the neural network

SAMPLE_RATE = 8000
SEGLEN = 100 # ms
BATCH_SIZE = 128

class Model(object):
	"""docstring for Model"""
	def __init__(self, path_to_model):
		# this should be of combined format as shown here
		# https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
		self.path_to_model = path_to_model

		# define the model
		self.model = models.load_model(path_to_model)

	def _resave(self):
		'''
		Function to save the model.
		'''
		self.model.save(self.path_to_model)

	def predict(self, input_segment, verbose = 0):
		'''
		Function to predict the class of the input chunk of audio
		Args:
			input_segment: segment to predict the class of audio
			verbose: verbosity parameter
		Returns:
			class_pred: predicted class of the input segment
		'''
		class_pred = self.model.predict(input_segment, verbose = verbose)
		return np.argmax(class_pred, axis = 1)

	def update(self, x, y, resave = True):
		'''
		Function that will train the model on a new examples and resave the model
		if told.
		Args:
			x: numpy array of input to the model
			y: numpy array of class output to the model
			resave: if True automatically reasave the model, else False
		'''
		self.model.train_on_batch(x, y)
		if resave:
			self._resave()

class Audio_Segmentation(object):
	"""
	docstring for Audio_Segmentation
	Args:
		audio: numpy array of loaded audio
		seglen: length of one chunk of audio (default 100 ms)
	"""
	def __init__(self, path_to_model):
		self.path_to_model = path_to_model

		# model
		self.model = Model(path_to_model)

	def _disp_audio_waveform(self, x, figsize = (30, 10)):
		'''
		Function to view the audio signal.
		Args:
			x: audio signal array
			figsize: size of plot created
		'''
		plt.figure(figsize = figsize)
		plt.plot(x)

	def _disp_audio_melspectrogram(self, x, figsize = (10, 10)):
		'''
		Function to view mfcc of input signal.
		Args:
			x: audio sample mfcc array
			figsize: size of plot created
		'''
		plt.figure(figsize = figsize)
		plt.imshow(x)

	def _convert_to_melspectrogram(self, x, sr):
		'''
		Return the audio chunk converted to mfcc
		'''
		return librosa.feature.melspectrogram(x, sr)

	def _segment_audio(self, x):
		'''
		Convert the input audio sample into segements
		'''
		# get 
		segsamples = int(SAMPLE_RATE*SEGLEN / 1000)
		num_samples = int((len(x) * 1000) / (SAMPLE_RATE * SEGLEN))
		# making data
		samples_ = []
		for i in range(num_samples):
			seg = x[i*segsamples: (i+1)*segsamples]
			samples_.append(seg)
		return samples_

	def view_segmentation(self):
		'''
		Function to view the segmented audio in form of plot
		'''
		pass

	def _pre_process(self, x, sr):
		'''
		Preprocess the audio
		'''
		x = librosa.core.to_mono(x)
		x = librosa.core.resample(x, sr, SAMPLE_RATE)
		return x

	def predict_audio(self, x, sr):
		# pre-process the data
		pp_ = self._pre_process(x, sr)

		# convert the audio in chunks
		samples_ = self._segment_audio(pp_)

		# get mfcc features for each sample
		melspectrogram_ = [self._convert_to_melspectrogram(i, sr) for i in samples_]

		# make batches
		if len(melspectrogram_) < BATCH_SIZE:
			diff = BATCH_SIZE - len(melspectrogram_)
			for _ in range(diff):
				melspectrogram_.append(melspectrogram_[0])

			# now we predict
			predictions = []
			for s in melspectrogram_:
				curr_batch = np.reshape(s, [BATCH_SIZE, 128, 2, 1])
				pred = self.model.predict(s)
				predictions.append(pred)

			return predictions[:diff]

		else:
			# case when input len is bigger than batch_size we can predict them

			num_batches = int(len(melspectrogram_) / BATCH_SIZE)
			diff = int(len(melspectrogram_) - (num_batches*BATCH_SIZE))
			for _ in range(diff):
				melspectrogram_.append(melspectrogram_[0])

			# now we predict
			predictions = []
			for i in range(num_batches):
				curr_batch = melspectrogram_[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
				curr_batch = np.reshape(curr_batch, [BATCH_SIZE, 128, 2, 1])
				pred = self.model.predict(curr_batch)
				predictions.extend(pred)

			return predictions
