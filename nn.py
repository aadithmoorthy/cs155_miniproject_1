# simple flat neural net
import numpy as np
import keras
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_split_data

Xtr, ytr, Xts, yts = get_split_data()
print ('loaded data')

# sequential model # 0.860 with this arch at 4 epochs (use without preprocessing)
model = Sequential()
model.add(Dense(1000, input_shape=(len(Xtr[0,:]),)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.9))
model.add(Dense(1))
model.add(Activation('sigmoid'))

checkpoint = ModelCheckpoint('nn_model.hdf5', monitor='val_acc', save_best_only=True, mode='max')
stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
model.fit(Xtr, ytr, batch_size=50, validation_data=(Xts, yts), epochs=100, verbose=1, callbacks=[checkpoint,stop])
