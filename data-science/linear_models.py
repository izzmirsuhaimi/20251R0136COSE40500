import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]  # Concatenate a column of ones to X for the intercept term

class LinearRegression:
    def __init__(self):
        self.weights = None

    def closed_form(self, X, y):
        X_b = add_bias(X)
        # Closed-form solution
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def gradient_descent(self, X, y, lr=0.01, epochs=1000):
        X_b = add_bias(X)
        m, n = X_b.shape
        self.weights = np.zeros(n)  # Initialize weights to zero
        for _ in range(epochs):
            gradients = (2 / m) * X_b.T @ (X_b @ self.weights - y)  # Compute gradient of MSE
            self.weights -= lr * gradients  # Update weights

    def predict(self, X):
        X_b = add_bias(X)
        return X_b @ self.weights  # Return predictions using learned weights

class LogisticRegression:
    def __init__(self):
        self.weights = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)  # Prevent overflow in exp calculation
        return 1 / (1 + np.exp(-z))  # Sigmoid function

    def fit(self, X, y, lr=0.01, epochs=1000):
        X_b = add_bias(X)
        m, n = X_b.shape
        self.weights = np.zeros(n)
        for _ in range(epochs):
            predictions = self.sigmoid(X_b @ self.weights)
            gradients = (1 / m) * X_b.T @ (predictions - y)  # Gradient of binary cross-entropy loss
            self.weights -= lr * gradients

    def predict_proba(self, X):
        X_b = add_bias(X)
        return self.sigmoid(X_b @ self.weights)  # Output probabilities

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)  # Convert probabilities to class labels (0 or 1)

if __name__ == '__main__':
    # Linear Regression on Diabetes Dataset
    diabetes = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Linear Regression - Closed Form
    custom_lr = LinearRegression()
    start = time.time()
    custom_lr.closed_form(X_train, y_train)
    time_closed = time.time() - start
    preds_custom_closed = custom_lr.predict(X_test)
    mse_closed = mean_squared_error(y_test, preds_custom_closed)

    # Custom Linear Regression - Gradient Descent
    custom_lr_gd = LinearRegression()
    start = time.time()
    custom_lr_gd.gradient_descent(X_train, y_train)
    time_gd = time.time() - start
    preds_custom_gd = custom_lr_gd.predict(X_test)
    mse_gd = mean_squared_error(y_test, preds_custom_gd)

    # Scikit-learn Linear Regression
    sklearn_lr = SklearnLinearRegression()
    start = time.time()
    sklearn_lr.fit(X_train, y_train)
    time_sklearn = time.time() - start
    preds_sklearn = sklearn_lr.predict(X_test)
    mse_sklearn = mean_squared_error(y_test, preds_sklearn)

    print("\nLinear Regression Results")
    print("-------------------------")
    print(f"Closed-form MSE: {mse_closed:.4f}, Time: {time_closed:.4f}s")
    print(f"Gradient Descent MSE: {mse_gd:.4f}, Time: {time_gd:.4f}s")
    print(f"Scikit-learn MSE: {mse_sklearn:.4f}, Time: {time_sklearn:.4f}s")

    # Compare weight coefficients
    print("\nWeight Coefficients Comparison (Linear Regression):")
    print("Custom (Closed-form):", custom_lr.weights)
    sklearn_weights = np.r_[sklearn_lr.intercept_, sklearn_lr.coef_]
    print("Scikit-learn:", sklearn_weights)

    # Logistic Regression on Breast Cancer Dataset
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2, random_state=42)

    scaler = StandardScaler()  # Standardization
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Custom Logistic Regression
    custom_logr = LogisticRegression()
    start = time.time()
    custom_logr.fit(X_train, y_train)
    time_logr = time.time() - start
    preds_logr = custom_logr.predict(X_test)
    acc_logr = accuracy_score(y_test, preds_logr)

    # Scikit-learn Logistic Regression
    sklearn_logr = SklearnLogisticRegression(max_iter=1000)
    start = time.time()
    sklearn_logr.fit(X_train, y_train)
    time_sklearn_logr = time.time() - start
    preds_sklearn_logr = sklearn_logr.predict(X_test)
    acc_sklearn_logr = accuracy_score(y_test, preds_sklearn_logr)

    print("\nLogistic Regression Results")
    print("---------------------------")
    print(f"Custom Accuracy: {acc_logr:.4f}, Time: {time_logr:.4f}s")
    print(f"Scikit-learn Accuracy: {acc_sklearn_logr:.4f}, Time: {time_sklearn_logr:.4f}s")

    # Compare weight coefficients
    print("\nWeight Coefficients Comparison (Logistic Regression):")
    print("Custom:", custom_logr.weights)
    sklearn_logr_weights = np.r_[sklearn_logr.intercept_, sklearn_logr.coef_.flatten()]
    print("Scikit-learn:", sklearn_logr_weights)
