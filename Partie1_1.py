import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Dropout, Softmax, MaxPooling2D, Flatten
from keras.datasets import mnist
from tensorflow.keras import backend as K
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# charger les donnees
(x_train, y_train),(x_test, y_test) = mnist.load_data()
img_rows = 28
img_cols = 28
num_classes = 10

# reshape a [echantillonage][pixels][largeur][longeuer]
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# normaliser les donnees et onehot les sorties
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# creation du modele et ajouter les couches
# Couches de convolution2D puis MaxPooling2D et Dropout pour reduire les bruits
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

# Compiler avec l'optimisation d'adam
model.compile(optimizer="adam", loss="categorical_crossentropy",metrics=["accuracy"])

# plotter les taux de validation et precision
hist = model.fit(x_train, y_train, epochs=12, validation_split=0.1)
plt.plot(hist.epoch, hist.history["accuracy"], label="Accuracy")
plt.plot(hist.epoch, hist.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()
accu = model.evaluate(x_test, y_test)

# afficher les taux de validation et precision
print("Taux d'erreur est",1-accu[1])
print("Taux d'accuracy est",accu[1])
