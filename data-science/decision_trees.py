import numpy as np
import time
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class DecisionTreeBase:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        return np.array([self._predict_input(x, self.tree) for x in X])

# Classification
class DecisionTreeClassifier(DecisionTreeBase):
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        current_entropy = self._entropy(y)

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] < t]
                right = y[X[:, feature] >= t]
                if len(left) == 0 or len(right) == 0:
                    continue
                p = float(len(left)) / len(y)
                gain = current_entropy - p * self._entropy(left) - (1 - p) * self._entropy(right)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return np.bincount(y).argmax()

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.bincount(y).argmax()

        indices_left = X[:, feature] < threshold
        left = self._build_tree(X[indices_left], y[indices_left], depth + 1)
        right = self._build_tree(X[~indices_left], y[~indices_left], depth + 1)
        return (feature, threshold, left, right)

    def _predict_input(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        return self._predict_input(x, left if x[feature] < threshold else right)

# Regression
class DecisionTreeRegressor(DecisionTreeBase):
    def _mse(self, y):
        if len(y) == 0:
            return 0
        return np.var(y) * len(y)

    def _best_split(self, X, y):
        best_mse = float("inf")
        best_feature, best_threshold = None, None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left = y[X[:, feature] < t]
                right = y[X[:, feature] >= t]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = self._mse(left) + self._mse(right)
                if mse < best_mse:
                    best_mse, best_feature, best_threshold = mse, feature, t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        if (depth >= self.max_depth or len(y) < self.min_samples_split):
            return np.mean(y)

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return np.mean(y)

        indices_left = X[:, feature] < threshold
        left = self._build_tree(X[indices_left], y[indices_left], depth + 1)
        right = self._build_tree(X[~indices_left], y[~indices_left], depth + 1)
        return (feature, threshold, left, right)

    def _predict_input(self, x, node):
        if not isinstance(node, tuple):
            return node
        feature, threshold, left, right = node
        return self._predict_input(x, left if x[feature] < threshold else right)

def run_experiments():
    # Classification
    print("Decision Tree Classifier (Breast Cancer Dataset)")
    print("------------------------------------------------")
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    for max_depth in [3, 5, 10]:
        for min_split in [2, 5, 10]:
            clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_split)
            start_time = time.time()
            clf.fit(X_train, y_train)
            end_time = time.time()
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            train_time = end_time - start_time
            print(f"depth={max_depth}, min_split={min_split} -> Accuracy: {acc}, Training Time: {train_time:.4f}s")

    # Regression
    print("\nDecision Tree Regressor (California Housing Dataset)")
    print("----------------------------------------------------")
    data = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

    for max_depth in [3, 5, 10]:
        for min_split in [2, 5, 10]:
            reg = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_split)
            start_time = time.time()
            reg.fit(X_train, y_train)
            end_time = time.time()
            y_pred = reg.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            train_time = end_time - start_time
            print(f"depth={max_depth}, min_split={min_split} -> MSE: {mse}, Training Time: {train_time:.4f}s")

if __name__ == "__main__":
    run_experiments()
