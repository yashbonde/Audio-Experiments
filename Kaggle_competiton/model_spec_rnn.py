'''
This is the RNN file to use the spectrograms.
'''
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM

PATH_SAVE_DATA = '../speech/pickles/spec_data.npy'
PATH_SAVE_OH = '../speech/pickles/spec_one_hot.npy'
PATH_SAVE_MODEL = '../speech/models/'

# loading the data
data = np.load(PATH_SAVE_DATA)
labels = np.load(PATH_SAVE_OH)

# paramters
input_shape = (81, 100, 1)

# model
model = Sequential()
model.add(LSTM(1024, input_shape = (81, 100)))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(12, activation = 'softmax'))
model.summary()

# compile
'''
using binary_crossentropy over categorical_crossentropy,
shot up the validation accuracy by about 9% from 89.5% to 98.5%
'''
model.compile(loss = 'binary_crossentropy',
	optimizer = 'Adam', metrics = ['accuracy'])

# fit the model
model.fit(x = data, y = labels, batch_size = 512,
	epochs = 25, validation_split = 0.2)

# saving the model
model.save(PATH_SAVE_MODEL + 'spec_rnn_.h5')
