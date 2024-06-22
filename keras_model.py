import keras
from keras import layers
import hickle as hkl

# Something about Concatenate
# Since Concatenate will remove axis whose value equals to 1, when using axis=2 in this case, you will get error.
# Because the original shape (None, 1, n, 128) will be changed to (None, n, 128), the axis=2 will refer to 128 instead of n and you will get a error
# In this case, you should use axis=-2 to avoid this problem

# DNA module
DNA_input = layers.Input(shape=(4, 300, 1))
DNA_conv1 = layers.Conv2D(128, (4, 8), padding='valid', activation='relu')(DNA_input)
DNA_conv2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(DNA_conv1)
DNA_conv3 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(DNA_conv2)
DNA_conv4 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(DNA_conv3)
DNA_pool1 = layers.MaxPool2D((1, 2))(DNA_conv4)
DNA_conv5 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(DNA_pool1)
DNA_conv6 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(DNA_conv5)

DNA_conv7 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(DNA_conv6)
DNA_pool2 = layers.MaxPool2D((1, 2))(DNA_conv7)

concat_DNA_conv2_conv3 = layers.Concatenate(axis=-1)([DNA_conv2, DNA_conv3])
concat_DNA_conv5_conv6 = layers.Concatenate(axis=-1)([DNA_conv5, DNA_conv6])
concat_DNA = layers.Concatenate(axis=-2)([DNA_conv1, concat_DNA_conv2_conv3, concat_DNA_conv5_conv6, DNA_pool2])

x = layers.Flatten()(concat_DNA)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
pred_output = layers.Dense(1, activation='sigmoid')(x)

DeepCAPE_only_DNA = keras.Model(DNA_input, pred_output, name='DeepCAPE_only_DNA')

# DNase module
DNase_input = layers.Input(shape=(1, 300, 1))
DNase_conv1 = layers.Conv2D(128, (1, 8), padding='valid', activation='relu')(DNase_input)
DNase_conv2 = layers.Conv2D(64, (1, 1), padding='same', activation='relu')(DNase_conv1)
DNase_conv3 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(DNase_conv2)
DNase_conv4 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(DNase_conv3)
DNase_pool1 = layers.MaxPool2D((1, 2))(DNase_conv4)
DNase_conv5 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(DNase_pool1)
DNase_conv6 = layers.Conv2D(64, (1, 3), padding='same', activation='relu')(DNase_conv5)

DNase_conv7 = layers.Conv2D(128, (1, 1), padding='same', activation='relu')(DNase_conv6)
DNase_pool2 = layers.MaxPool2D((1, 2))(DNase_conv7)

concat_DNase_conv2_conv3 = layers.Concatenate(axis=-1)([DNase_conv2, DNase_conv3])
concat_DNase_conv5_conv6 = layers.Concatenate(axis=-1)([DNase_conv5, DNase_conv6])
concat_DNase = layers.Concatenate(axis=-2)([DNase_conv1, concat_DNase_conv2_conv3, concat_DNase_conv5_conv6, DNase_pool2])

x = layers.Flatten()(concat_DNase)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
pred_output = layers.Dense(1, activation='sigmoid')(x)

DeepCAPE_only_DNase = keras.Model(DNase_input, pred_output, name='DeepCAPE_only_DNase')

# DeepCAPE
DNA_pool2 = layers.MaxPool2D((1, 2))(DNA_conv6)
DNase_pool2 = layers.MaxPool2D((1, 2))(DNase_conv6)
concat_DNA_DNase_pool2 = layers.Concatenate(axis=-1)([DNA_pool2, DNase_pool2])
concat_DNA_DNase = layers.Concatenate(axis=-2)([DNA_conv1, concat_DNA_conv2_conv3, concat_DNA_conv5_conv6, concat_DNA_DNase_pool2, concat_DNase_conv5_conv6, concat_DNase_conv2_conv3, DNase_conv1])

x = layers.Flatten()(concat_DNA_DNase)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
pred_output = layers.Dense(1, activation='sigmoid')(x)

DeepCAPE = keras.Model([DNA_input, DNase_input], pred_output, name='DeepCAPE')


DeepCAPE_only_DNA.summary()
