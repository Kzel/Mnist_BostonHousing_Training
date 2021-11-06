import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Dropout, Softmax
from keras.layers.advanced_activations import LeakyReLU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

(train_xdata, train_ydata), (test_xdata, test_ydata) = keras.datasets.boston_housing.load_data()
# 2.1.1
train_xdata_norm = train_xdata
x_moy = np.mean(train_xdata,axis=0)
train_xdata_norm -= x_moy
x_std = np.std(train_xdata,axis=0)
train_xdata_norm /= x_std

print(np.mean(train_xdata_norm,axis=0))
print(np.std(train_xdata_norm,axis=0))

# 2.1.2
m = Sequential()
m.add(Dense(20, input_shape=(13,)))
#m.add(Dense(20, activation="relu",input_shape=(13,)))
m.add(LeakyReLU(alpha=0.2))
m.add(Dense(1))
#m.add(Dense(1, activation="relu"))
m.add(LeakyReLU(alpha=0.2))
m.summary()

# 2.1.3
m.compile(optimizer="adam", loss="mean_squared_error")
history = m.fit(train_xdata_norm, train_ydata, epochs=400 ,validation_split=0.15)
plt.plot(history.epoch, history.history["loss"], label="Training error")
plt.plot(history.epoch, history.history["val_loss"], label="Validation error")
plt.legend()
plt.show()

test_xdata_norm = test_xdata
testx_moy = np.mean(test_xdata,axis=0)
test_xdata_norm -= testx_moy
testx_std = np.std(test_xdata,axis=0)
test_xdata_norm /= testx_std
erreur =  m.evaluate(test_xdata_norm , test_ydata)
print("Erreur est:",np.sqrt(erreur))