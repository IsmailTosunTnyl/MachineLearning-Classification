from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from Utils import RawData, Plotter

# Taking clean data from RawData() class
df = RawData().df

# Separating features and label
X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable

# Separating train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Initializing SVM modals with rbf algorithm
clf = svm.SVC(kernel='rbf')  # ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’

# Applying standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model with train sets
clf.fit(X_train, y_train)

# Testing model with test sets
y_pred = clf.predict(X_test)

# Printing accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot Common Graphs
Plotter().plot_cofusion_matrix(y_test, y_pred, "Support Vector Machine")
Plotter().plot_traning_curves(X, y, clf, "Support Vector Machine")
