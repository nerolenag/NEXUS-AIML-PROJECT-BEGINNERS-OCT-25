import numpy as np
import pandas as pd

# ======================
# K-NEAREST NEIGHBORS
# ======================
class KNearestNeighborsScratch:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def predict(self, X):
        X = np.array(X)
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        values, counts = np.unique(k_nearest_labels, return_counts=True)
        return values[np.argmax(counts)]


# ======================
# EXAMPLE: Simple KNN Classification
# ======================
data = pd.DataFrame({
    'x1': [1, 2, 3, 6, 7, 8],
    'x2': [1, 1, 2, 6, 7, 8],
    'label': [0, 0, 0, 1, 1, 1]
})

X = data[['x1', 'x2']]
y = data['label']

# Initialize and train model
knn = KNearestNeighborsScratch(k=3)
knn.fit(X, y)

# Predict for new samples
new_samples = pd.DataFrame({
    'x1': [4, 5, 7],
    'x2': [3, 4, 8]
})

predictions = knn.predict(new_samples)
print("KNN Predictions for new samples:", predictions)
