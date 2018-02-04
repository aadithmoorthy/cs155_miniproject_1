# simple flat neural net
import numpy as np
import keras
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense, Activation, Flatten, Dropout
import matplotlib.pyplot as plt
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from sklearn.preprocessing import StandardScaler


Xtr, ytr, Xts, yts = get_split_data()
trunc = 500
Xtr = Xtr[:,:trunc] #truncation
Xts = Xts[:,:trunc] #truncation
print ('loaded data')



'''
# sequential model # 0.857 with this arch at 4 epochs (use without preprocessing) - saved in nn_model1.h5 (scored 0.852 in leaderboard!)
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
'''

fit_model = True
if fit_model:
    model = Sequential()
    model.add(Dense(trunc, input_shape=(len(Xtr[0,:]),)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    checkpoint = ModelCheckpoint('nn_model.hdf5', monitor='val_acc', save_best_only=True, mode='max')
    stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    model.fit(Xtr, ytr, batch_size=50, validation_data=(Xts, yts), epochs=100, verbose=1, callbacks=[checkpoint,stop])

else:
    # predict
    model = load_model('nn_model1.hdf5')

    result_col_2 = model.predict(get_test_data())

    result_col_2 = result_col_2.reshape((len(result_col_2),1))
    result_col_1 = (np.array(range(len(result_col_2)))+1).reshape((len(result_col_2),1))
    result = np.concatenate((result_col_1,result_col_2), axis = 1)
    np.savetxt('nn_result.txt', np.round(result), fmt="%d", delimiter=',', header='Id,Prediction')

    result = model.predict(get_test_data())
    np.savetxt('nn_result_probabilities.txt', result, fmt="%g", delimiter=',')

    result = model.predict(Xts)
    print accuracy_score(yts, np.round(result))
    np.savetxt('nn_merge_result.txt', result, fmt="%g", delimiter=',')
