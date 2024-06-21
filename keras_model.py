from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Concatenate

input_seq  = Input(shape=(4, 300, 1))
seq_conv1_ = Conv2D(128, (4, 8), activation='relu',padding='valid')
seq_conv1  = seq_conv1_(input_seq)
seq_conv2_ = Conv2D(64, (1, 1), activation='relu',padding='same')
seq_conv2  = seq_conv2_(seq_conv1)
seq_conv3_ = Conv2D(64, (1, 3), activation='relu',padding='same')
seq_conv3  = seq_conv3_(seq_conv2)
seq_conv4_ = Conv2D(128, (1, 1), activation='relu',padding='same')
seq_conv4  = seq_conv4_(seq_conv3)
seq_pool1  = MaxPooling2D(pool_size=(1, 2))(seq_conv4)
seq_conv5_ = Conv2D(64, (1, 3), activation='relu',padding='same')
seq_conv5  = seq_conv5_(seq_pool1)
seq_conv6_ = Conv2D(64, (1, 3), activation='relu',padding='same')
seq_conv6  = seq_conv6_(seq_conv5)
#
seq_conv7_ = Conv2D(128, (1, 1), activation='relu',padding='same')
seq_conv7  = seq_conv7_(seq_conv6)
#
seq_pool2  = MaxPooling2D(pool_size=(1, 2))(seq_conv7)
merge_seq_conv2_conv3 = Concatenate(axis=-1)([seq_conv2, seq_conv3])
merge_seq_conv5_conv6 = Concatenate(axis=-1)([seq_conv5, seq_conv6])
x = Concatenate(axis=2)([seq_conv1, merge_seq_conv2_conv3, merge_seq_conv5_conv6, seq_pool2])
x = Flatten()(x)
dense1_ = Dense(512, activation='relu')
dense1  = dense1_(x)
dense2  = Dense(256, activation='relu')(dense1)
x = Dropout(0.5)(dense2)
dense3 = Dense(128, activation='relu')(x)
pred_output = Dense(1, activation='sigmoid')(dense3)

# Keras model
model = Model(inputs=[input_seq], outputs=[pred_output])