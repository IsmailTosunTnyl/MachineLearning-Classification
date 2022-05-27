import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

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


def traning_curves():
    global X, y, lr
    train_sizes = [1, 100, 500, 2000, 5000, 7654, 10000, 15000, 20000, 22000]
    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=lr,
        X=X,
        y=y, train_sizes=train_sizes, cv=5,
        scoring='neg_mean_squared_error')

    train_scores_mean = -train_scores.mean(axis=1)
    validation_scores_mean = -validation_scores.mean(axis=1)

    plt.style.use('seaborn')
    plt.plot(train_sizes, train_scores_mean, label='Training error')
    plt.plot(train_sizes, validation_scores_mean, label='Validation error')
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Training set size', fontsize=14)
    plt.title('Learning curves for a Logistic Regression model', fontsize=18, y=1.03)
    plt.legend()
    plt.ylim(0, 40)
    plt.show()


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
traning_curves()
