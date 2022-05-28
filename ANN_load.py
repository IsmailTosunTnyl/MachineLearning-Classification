import numpy as np
from keras.saving.save import load_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

# Load pretrained model
ann = load_model('ann_50_50.h5')

#   Use model
y_pred = ann.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
Utils.Plotter().plot_cofusion_matrix(y_test, y_pred, "ANN")
