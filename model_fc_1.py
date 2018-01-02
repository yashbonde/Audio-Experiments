'''
This is the machine learning file:
We do changes to the standard feed-forward model here.
Thus saving the time for prototyping
'''
import numpy as np # matrix math
import os # interaction with os

from keras.models import Sequential
from keras.layers import Dense, Dropout

PATH_SAVE_DATA = '/home/bonde_yash97/speech/pickles/data.npy'
PATH_SAVE_OH = '/home/bonde_yash97/speech/pickles/one_hot.npy'
PATH_SAVE_MODEL = '/home/bonde_yash97/speech/'

# loading the data
data = np.load(PATH_SAVE_DATA)
labels = np.load(PATH_SAVE_OH)

# parameters
# learning_rate = 0.01
n_hidden_1 = 1024
n_hidden_2 = 512
n_hidden_3 = 256

# defining the model
model = Sequential()
model.add(Dense(n_hidden_1, input_shape = (193,)))
# model.add(Dropout(0.1))
model.add(Dense(n_hidden_2))
# model.add(Dropout(0.2))
model.add(Dense(n_hidden_3))
model.add(Dense(12))
print(model.summary())

model.compile(optimizer = 'rmsprop',
	loss = 'categorical_crossentropy',
	metrics = ['accuracy'])

# training the model
model.fit(x = data,
	y = labels,
	batch_size = 512,
	epochs = 5000,
	validation_split = 0.2,
	verbose = 2)

# saving the model once trained
model.save(PATH_SAVE_MODEL + 'model_fc_1.h5')
