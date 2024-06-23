import keras
from keras import callbacks
from keras import optimizers

from keras_model import DeepCAPE_only_DNA
from preprocess import Preprocessor

import numpy as np
from sklearn import metrics

# preprocess data
preprocess = Preprocessor()
X_train, X_test, y_train, y_test = preprocess.generateSamples()

# train model
early_stopping = callbacks.EarlyStopping(monitor='val_loss', verbose=0, patience=3, mode='min')
save_best = callbacks.ModelCheckpoint('./processed_data/model_DNA_only.weights.h5', save_best_only=True, save_weights_only=True)
adam = optimizers.Adam(learning_rate=1e-4, epsilon=1e-08, weight_decay=1e-6)
DeepCAPE_only_DNA.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
DeepCAPE_only_DNA.fit(X_train, y_train, batch_size=128, epochs=30, validation_split=0.1, callbacks=[early_stopping, save_best])

# predict
y_pred = DeepCAPE_only_DNA.predict(X_test)
auROC = metrics.roc_auc_score(y_test, y_pred)
print('auROC = {}'.format(auROC))
auPR = metrics.average_precision_score(y_test, y_pred)
print('auPR  = {}'.format(auPR))