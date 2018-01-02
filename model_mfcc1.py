'''
Model Code:
Using MFCC Dump
'''

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

PATH_SAVE_DATA_1 = '/home/bonde_yash97/speech/pickles/mfcc_data_1.npy'
PATH_SAVE_DATA_2 = '/home/bonde_yash97/speech/pickles/mfcc_data_2.npy'
PATH_SAVE_OH = '/home/bonde_yash97/speech/pickles/mfcc_one_hot.npy'
PATH_SAVE_MODEL = '/home/bonde_yash97/speech/'

# loading the data
print('[!]Loading data')
data = np.load(PATH_SAVE_DATA_2)
labels = np.load(PATH_SAVE_OH)

# params
input_shape = (98, 40, 1)
val_split = 0.2

# splitting of data
train_data = data[:-int(val_split*len(data))]
train_label = labels[:-int(val_split*len(data))]
val_data = data[int((1-val_split) * len(data)):]
val_labels = labels[int((1-val_split) * len(data)):]

print('[*]train_data.shape:', train_data.shape)
print('[*]train_label.shape:', train_label.shape)
print('[*]val_data.shape:', val_data.shape)
print('[*]val_labels.shape:', val_labels.shape)

# model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = input_shape))
model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.25))
model.add(Dense(12))
print(model.summary())

# compiling
model.compile(loss = 'categorical_crossentropy', optimizer = 'Adam', metrics = ['accuracy'])

# fitting the data
model.fit(x = train_data, y = train_label, batch_size = 512, epochs = 15, validation_data = (val_data, val_labels))