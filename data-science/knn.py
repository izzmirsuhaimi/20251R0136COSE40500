import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class KNearestNeighbors:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.array([self.compute_distance(x, x_train) for x_train in self.X_train])
            k_indices = distances.argsort()[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            unique, counts = np.unique(k_nearest_labels, return_counts=True)
            prediction = unique[counts.argmax()]
            predictions.append(prediction)
        return np.array(predictions)

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.neighbors import KNeighborsClassifier

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k_values = [3, 5, 7, 9]
    metrics = ['euclidean', 'manhattan']

    print("\nCustom KNN Results:")
    for metric in metrics:
        for k in k_values:
            knn_custom = KNearestNeighbors(k=k, distance_metric=metric)
            knn_custom.fit(X_train, y_train)
            predictions_custom = knn_custom.predict(X_test)
            accuracy_custom = accuracy_score(y_test, predictions_custom)
            print(f"Custom KNN (k={k}, metric={metric}): Accuracy = {accuracy_custom:.4f}")

    print("\nscikit-learn KNN Results:")
    for metric in metrics:
        for k in k_values:
            knn_sklearn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn_sklearn.fit(X_train, y_train)
            predictions_sklearn = knn_sklearn.predict(X_test)
            accuracy_sklearn = accuracy_score(y_test, predictions_sklearn)
            print(f"scikit-learn KNN (k={k}, metric={metric}): Accuracy = {accuracy_sklearn:.4f}")

    # t-SNE Visualization
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    plt.title("t-SNE Visualization of Iris Dataset")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
