import numpy as np
import pandas as pd
from keras.backend import flatten
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

import Utils
from Utils import RawData

df = RawData().df
X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

ann = tf.keras.models.Sequential()


y_train = np_utils.to_categorical(y_train)

ann.add(tf.keras.layers.Dense(units=20, activation="swish")) #tanh relu selu
#
ann.add(tf.keras.layers.Dense(units=20, activation="swish"))
#ann.add(tf.keras.layers.Dense(units=8, activation="relu"))
#ann.add(tf.keras.layers.Dense(units=8, activation="relu"))

ann.add(tf.keras.layers.Dense(units=10, activation="softmax"))
ann.compile(optimizer="Nadam", loss="categorical_crossentropy", metrics=['accuracy']) #SGD Nadam Adam RMSprop
ann.fit(X_train, y_train, batch_size=16, epochs=100)
y_pred = ann.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
Utils.Plotter().plot(y_test, ypred)