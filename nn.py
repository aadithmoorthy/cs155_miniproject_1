# simple flat neural net
import numpy as np
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop

from utils import get_split_data

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')




# sequential model # 0.857 with this arch at 12 epochs
'''model = Sequential()
model.add(Dense(1000, input_shape=(len(Xtr[0,:]),)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.9))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
model.fit(Xtr, ytr, batch_size=50, validation_data=(Xts, yts), epochs=20, verbose=1)
'''
model = Sequential()
model.add(Dense(1000, input_shape=(len(Xtr[0,:]),)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1000, input_shape=(len(Xtr[0,:]),)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.8))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=RMSprop(lr=0.0002), metrics=['accuracy'])
model.fit(Xtr, ytr, batch_size=128, validation_data=(Xts, yts), epochs=50, verbose=1)
