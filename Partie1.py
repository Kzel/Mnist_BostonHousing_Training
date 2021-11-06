import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, Dropout, Softmax
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 1.1.1
x_data = np.array([[-1.,-1.],
                   [-1.,1.],
                   [1.,-1.],
                   [1.,1.]])
y_data = np.array([[-1.],
                   [1.],
                   [1.],
                   [-1.]])

print(x_data.shape) # (4,2): 4 exemples d'apprentissage en 2 dimensions
print(y_data.shape) # (4,1): 4 sorites desirees en 1 dimension

# 1.1.2
m = Sequential()

m.add(Dense(3, activation="tanh", input_shape=(2,)))
m.add(Dense(1, activation="tanh"))
m.summary()
keras.utils.plot_model(m, show_shapes=True)

# 1.1.3
opti_sgd005 = keras.optimizers.SGD(learning_rate=0.05)
m.compile(optimizer=opti_sgd005, loss="mean_squared_error")

# 1.2
m.predict(x_data)
hist = m.fit(x_data, y_data, epochs=500)
plt.plot(hist.epoch, hist.history["loss"])
plt.show()
