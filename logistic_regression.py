# Logistic Regression 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(0)
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train_logistic(X, y, lr=0.1, epochs=1000):
    m, n = X.shape
    weights = np.zeros(n)
    bias = 0
    for _ in range(epochs):
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
        dw = (1/m) * np.dot(X.T, (y_pred - y))
        db = (1/m) * np.sum(y_pred - y)
        weights -= lr * dw
        bias -= lr * db
    return weights, bias

weights, bias = train_logistic(X_train, y_train)

def predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return sigmoid(z) >= 0.5

y_pred = predict(X_test, weights, bias)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

with open("saved_models/logistic_model.pkl", "wb") as f:
    pickle.dump((weights, bias), f)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k')
x_values = np.linspace(-3, 3, 100)
y_values = -(weights[0] / weights[1]) * x_values - bias / weights[1]
plt.plot(x_values, y_values, color='black')
plt.title('Logistic Regression')
plt.savefig("screenshots/logistic_regression_plot.png")
plt.show()
