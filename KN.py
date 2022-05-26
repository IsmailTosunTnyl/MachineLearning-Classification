import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from Utils import RawData, Plotter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = RawData().df

X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable
# scaling
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(X)
data = pd.DataFrame(X_data_minmax, columns=RawData().feature_cols)
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.25, random_state=1)
print(data.head())

knn_clf = KNeighborsClassifier(n_neighbors=16)
knn_clf.fit(X_train, y_train)
ypred = knn_clf.predict(X_test)

print(accuracy_score(y_test, ypred))
Plotter().plot(y_test, ypred)

scaler = StandardScaler()
df = RawData().df
X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

knn_clf.fit(X_train_s, y_train)
ypred = knn_clf.predict(X_test_s)

print(accuracy_score(y_test, ypred))

# k finder
error = list()
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train_s, y_train)
    pred_i = knn.predict(X_test_s)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
