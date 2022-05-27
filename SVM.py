from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

import Utils
from Utils import RawData

df = RawData().df

X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

clf = svm.SVC(kernel='rbf')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

Utils.Plotter().plot(y_test,y_pred)
Utils.Plotter().traning_curves(X,y,clf,"Support Vector Machine")
