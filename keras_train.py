import hickle as hkl
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers.legacy import Adam

from keras_model import model
from preprocess import Preprocessor

# preprocess data
preprocess = Preprocessor()
preprocess.generateSamples()

# create dataset
train_X_seq, train_y = hkl.load(preprocess.train_sample_file_paths[0])
train_X_seq = train_X_seq.reshape(-1, 4, 300, 1)
train_y = train_y.reshape(-1, 1)
indice = np.arange(train_y.shape[0])
np.random.shuffle(indice)
train_X_seq  = train_X_seq[indice]
train_y = train_y[indice]

# train model
early_stopping = EarlyStopping(monitor='val_loss', verbose=0, patience=3, mode='min')
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
model.fit(train_X_seq, train_y, batch_size=128, epochs=30, validation_split=0.1, callbacks=[early_stopping])