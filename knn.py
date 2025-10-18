# K-Nearest Neighbors
import numpy as np
import pickle
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, X_test, k=3):
    y_pred = []
    for test_point in X_test:
        distances = [euclidean_distance(test_point, x) for x in X_train]
        k_idx = np.argsort(distances)[:k]
        k_neighbor_labels = [y_train[i] for i in k_idx]
        most_common = Counter(k_neighbor_labels).most_common(1)[0][0]
        y_pred.append(most_common)
    return np.array(y_pred)

y_pred = knn_predict(X_train, y_train, X_test)
acc = accuracy_score(y_test, y_pred)
print("KNN Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("saved_models/knn_model.pkl", "wb") as f:
    pickle.dump((X_train, y_train), f)
