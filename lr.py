from sklearn.linear_model import LogisticRegression

from Utils import RawData,Plotter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
df = RawData().df

X = df[RawData().feature_cols]  # Features
y = df.music_genre  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

lr = LogisticRegression(max_iter=1000000)

lr.fit(X_train, y_train)

pred = lr.predict(X_test)

print(lr.score(X_test, y_test))
print(accuracy_score(y_test, pred))

# standardization
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s    = scaler.transform(X_test)

lr.fit(X_train_s, y_train)
pred = lr.predict(X_test_s)

print(lr.score(X_test_s, y_test))
print(accuracy_score(y_test, pred))
Plotter().plot(y_test,pred)