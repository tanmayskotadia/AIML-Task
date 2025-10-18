# Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_b_train = np.c_[np.ones((len(X_train), 1)), X_train]
X_b_test = np.c_[np.ones((len(X_test), 1)), X_test]

theta_best = np.linalg.inv(X_b_train.T.dot(X_b_train)).dot(X_b_train.T).dot(y_train)

print("Learned parameters (theta):", theta_best.ravel())

y_pred = X_b_test.dot(theta_best)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Test MSE:", mse)
print("Test R2:", r2)

plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.savefig("screenshots/linear_regression_plot.png")
plt.show()

with open("saved_models/linear_model.pkl", "wb") as f:
    pickle.dump(theta_best, f)
