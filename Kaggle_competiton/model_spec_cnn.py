'''
This is a CNN model, it was built to use raw spectrograms.
'''
# importing dependencies
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

PATH_SAVE_DATA = '../speech/pickles/spec_data.npy'
PATH_SAVE_OH = '../speech/pickles/spec_one_hot.npy'
PATH_SAVE_MODEL = '../speech/'

# loading the data
data = np.load(PATH_SAVE_DATA)
labels = np.load(PATH_SAVE_OH)

# reshaping the data for CNN
s = data.shape
data = np.reshape(data, [s[0], s[1], s[2], 1])

# reshaping the data for CNN
s = data.shape
data = np.reshape(data, [s[0], s[1], s[2], 1])

# paramters
input_shape = (81, 100, 1)

# model
model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), activation = 'relu',
	input_shape = (81, 100, 1)))
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (3, 3)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (2, 2), activation = 'relu'))
model.add(Conv2D(128, kernel_size = (2, 2), activation = 'relu'))
model.add(Conv2D(128, kernel_size = (2, 2), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
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
# model.save(PATH_SAVE_MODEL + 'spec_cnn_1_.h5')
