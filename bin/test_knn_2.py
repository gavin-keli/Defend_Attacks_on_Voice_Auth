import pandas as pd
from sklearn.model_selection  import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("../data/guardian/knn_file/5651345512-100_train1.csv")
data2 = pd.read_csv("../data/guardian/knn_file/5651345512-100_test1.csv")

x = data.iloc[:,1:]
y = data.iloc[:,0:1]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

n_neighbors = 11
neigh = KNeighborsClassifier(n_neighbors)
neigh.fit(X_train, y_train)

neigh2 = KNeighborsClassifier(n_neighbors, weights='distance')
neigh2.fit(X_train, y_train)
y_pred = neigh2.predict(x)
print(confusion_matrix(y,y_pred))
print(classification_report(y,y_pred))

x2 = data2.iloc[:,1:]
y2 = data2.iloc[:,0:1]

y_pred2 = neigh2.predict(x2)
print(confusion_matrix(y2,y_pred2))
print(classification_report(y2,y_pred2))