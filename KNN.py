import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from Utils import RawData, Plotter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Taking clean data from RawData() class
df = RawData().df

# Separating features and label
X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable

# Separating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Initializing KNeighborsClassifier modals with 16 neighbors hyperparameters algorithm
knn_clf = KNeighborsClassifier(n_neighbors=16)

# Applying standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model with train sets
knn_clf.fit(X_train, y_train)

# Testing model with test sets
y_pred = knn_clf.predict(X_test)

# Printing accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot Common Graphs
Plotter().plot_cofusion_matrix(y_test, y_pred, "K-Nearest Neighbors")
Plotter().plot_traning_curves(X, y, knn_clf, "K-Nearest Neighbors")

# K finder
# This algorithm try different n_neighbors values and plot a diagram
error = list()
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()
