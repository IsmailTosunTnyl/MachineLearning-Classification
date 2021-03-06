import graphviz as graphviz

import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
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

# Initializing DecisionTreeClassifier models with  hyperparameters
clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=12)  # hyperparameters

# Applying standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model with train sets
clf = clf.fit(X_train, y_train)

# Testing model with test sets
y_pred = clf.predict(X_test)

# Printing accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot Common Graphs
Plotter().plot_cofusion_matrix(y_test, y_pred, "Decision Tree")
Plotter().plot_traning_curves(X, y, clf, "Decision Tree")

# Drawing a leaf graph it may not fit to pdf if you want to see all tree use smaller max_depth values
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data, filename="dt.gv", format="pdf")
graph.render("DT")
graph.save()

# This algorithm try different max_depht values and plot a diagram
error = list()
for i in range(1, 40):
    clf = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=i)
    clf.fit(X_train, y_train)
    pred_i = clf.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate max depth Value')
plt.xlabel('Max Depth')
plt.ylabel('Mean Error')
plt.show()
