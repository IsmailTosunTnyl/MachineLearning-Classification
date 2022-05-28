import numpy as np
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from Utils import RawData, Plotter

# Taking clean data from RawData() class
df = RawData().df

# Separating features and label
X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable

# Separating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Applying standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing ANN models
ann = tf.keras.models.Sequential()

# Reshaping y_train for ANN algorith
y_train = np_utils.to_categorical(y_train)

# Adding hidden layers
# We use 2 hidden layers with 30 units ans swish activation functions
ann.add(tf.keras.layers.Dense(units=30, activation="swish"))  # tanh relu selu
ann.add(tf.keras.layers.Dense(units=30, activation="swish"))

# Adding last layer with softmax activator function and 10 units because we have 10 different category
ann.add(tf.keras.layers.Dense(units=10, activation="softmax"))

# Compiling ANN model
ann.compile(optimizer="Nadam", loss="categorical_crossentropy", metrics=['accuracy'])  # SGD Nadam Adam RMSprop

# Traninig model and save history for plot graphs
# batch_size, epochs, validation_split are our hyperparameters
history = ann.fit(X_train, y_train, batch_size=20, epochs=50, validation_split=0.25)

# Saving trained model as file, we can use later this model
ann.save("ann_30_100.h5")

# Making prediction
y_pred = ann.predict(X_test)

# Restoring array for drawing graphs and calculating accuracy
y_pred = np.argmax(y_pred, axis=1)

# Calculating accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot common graph
Plotter().plot_cofusion_matrix(y_test, y_pred, "ANN")

# Plot accuracy/validation accuracy - epoch graph
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plot loss/validation loss - epoch graph
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
