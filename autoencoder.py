# autoencoder

from keras.layers import Dense, Input, BatchNormalization, Dropout, Activation
from keras.models import load_model
from keras.models import Model, Sequential

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import RMSprop
from utils import get_split_data
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

Xtr, ytr, Xts, yts = get_split_data()

train = False
if train:
    num_encoding = 128 # 0.0180 val loss with 128 (1000-128-1000) and lr=0.00001 stopped after 100 epochs (wihout preprocessing)

    inp = Input(batch_shape=(50,1000))
    # 1 layer of 1000
    hidden_1_enc = Dense(1000, activation='relu')(inp)

    encoding = Dense(num_encoding)(hidden_1_enc)

    # for decoding
    # 1 layers of 1000
    # do we need this layer even?
    hidden_1_dec = Dense(1000, activation='relu')(encoding)

    # we have non-negative values in the data so relu can be used (linear for preprocessed)
    outp = Dense(1000, activation='relu')(hidden_1_dec)

    model_enc_dec = Model(inp, outp)

    model_enc_dec.compile(optimizer='rmsprop', loss='mse')
    checkpoint = ModelCheckpoint('autoencoder.hdf5', monitor='val_loss', save_best_only=True)
    #checkpoint = EarlyStopping(monitor='val_loss', patience=10)
    model_enc_dec.fit(Xtr, Xtr, batch_size=50, validation_data=(Xts, Xts), epochs=1000, verbose=1, callbacks=[checkpoint])

else:
    # test with logistic regression as bench mark:
    autoencoder_decoder = load_model('autoencoder.hdf5')
    autoencoder = Model(inputs=autoencoder_decoder.input, outputs=autoencoder_decoder.layers[2].output)
    Xtr_transform = autoencoder.predict(Xtr, batch_size=50)
    Xts_transform = autoencoder.predict(Xts, batch_size=50)
    print autoencoder_decoder.predict(Xtr, batch_size=50)
    print Xtr
    model = Sequential()
    model.add(Dense(512, input_shape=(len(Xtr_transform[0,:]),)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.8))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    #checkpoint = ModelCheckpoint('nn_model.hdf5', monitor='val_acc', save_best_only=True, mode='max')
    stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['accuracy'])
    model.fit(Xtr_transform, ytr, batch_size=50, validation_data=(Xts_transform, yts), epochs=100, verbose=1, callbacks=[stop])
