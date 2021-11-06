import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Dropout, Softmax
from keras.layers.advanced_activations import LeakyReLU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
(train_xdata, train_ydata), (test_xdata, test_ydata) = keras.datasets.mnist.load_data()

#plt.imshow(train_xdata[0])
#plt.show()
#print(train_xdata.shape)

n_pixels = train_xdata.shape[1] * train_xdata.shape[2]
train_xdata_flat = train_xdata.reshape(train_xdata.shape[0], n_pixels)
test_xdata_flat = test_xdata.reshape(test_xdata.shape[0], n_pixels)

train_xdata_flat = train_xdata_flat.astype("float32") / 255
test_xdata_flat = test_xdata_flat.astype("float32") / 255

train_ydata_onehot = keras.utils.to_categorical(train_ydata)
test_ydata_onehot = keras.utils.to_categorical(test_ydata)

#3.1.2
m = Sequential()
m.add(Dense(784, activation="relu", input_shape=(784,)))
m.add(Dense(100, activation="relu"))
m.add(Dense(100, activation="relu"))
m.add(Dense(10, activation="softmax"))
m.summary()

#3.1.3
m.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])

#3.2
hist = m.fit(train_xdata_flat, train_ydata_onehot, epochs=15 ,validation_split=0.1)
plt.plot(hist.epoch, hist.history["accuracy"], label="Accuracy")
plt.plot(hist.epoch, hist.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()

#3.3
accu = m.evaluate(test_xdata_flat,test_ydata_onehot)
print("Taux d'erreur est",1-accu[1])
print("Taux d'accuracy est",accu[1])