import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

import Utils
from Utils import RawData, Plotter
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = RawData().df

X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# todo iter error graph
lr = LogisticRegression(max_iter=len(X_train))

# standardization
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

Plotter().plot(y_test, y_pred)



def iter_accuracy():
    global X_train,X_test
    error = list()
    for i in range(1, 100,2):
        lr = LogisticRegression(max_iter=i)

        lr.fit(X_train, y_train)

        pred_i = lr.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, 100,2), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate iteration Value')
    plt.xlabel('iteration Value')
    plt.ylabel('Mean Error')
    plt.show()

iter_accuracy()
Utils.Plotter().traning_curves(X,y,lr,"Logistic Regression")
