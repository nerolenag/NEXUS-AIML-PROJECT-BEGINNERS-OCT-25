import numpy as np
import pandas as pd

# ==============================
# LOGISTIC REGRESSION FROM SCRATCH
# ==============================
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        X = np.array(X)
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.where(y_pred >= 0.5, 1, 0)


# ==============================
# EXAMPLE: Simple Binary Classification
# ==============================
data = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5, 6],
    'feature2': [2, 1, 2, 3, 2, 3],
    'label':    [0, 0, 0, 1, 1, 1]
})

X = data[['feature1', 'feature2']]
y = data['label']

# Train model
model = LogisticRegressionScratch(learning_rate=0.1, epochs=1000)
model.fit(X, y)

# Predict on new data samples
new_samples = pd.DataFrame({
    'feature1': [2, 4, 6],
    'feature2': [1, 2, 3]
})

predictions = model.predict(new_samples)
print("Predictions for new samples:", predictions)
