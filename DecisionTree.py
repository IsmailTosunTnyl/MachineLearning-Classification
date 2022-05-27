import graphviz as graphviz

import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Utils import RawData

df = RawData().df
X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

clf = tree.DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=12)
clf = clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data,filename="dt.gv",format="pdf")
graph.render("iris")
graph.save()

"""error = list()
for i in range(1, 40):
    clf = tree.DecisionTreeClassifier(criterion="gini",splitter="best",max_depth=i)
    clf.fit(X_train, y_train)
    pred_i = clf.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()"""