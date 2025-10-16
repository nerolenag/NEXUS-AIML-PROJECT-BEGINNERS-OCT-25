import numpy as np
import pandas as pd

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([0, 0, 0, 1, 1])

w = np.zeros((X.shape[1],))
b = 0
lr = 0.1
epochs = 1000

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


n = len(y)
for i in range(epochs):
    z = np.dot(X, w) + b
    y_pred = sigmoid(z)
    
    dw = (1/n) * np.dot(X.T, (y_pred - y))
    db = (1/n) * np.sum(y_pred - y)
    
    w -= lr * dw
    b -= lr * db

print(f"Weights: {w}")
print(f"Bias: {b:.2f}")


def predict(X):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return [1 if p > 0.5 else 0 for p in probs]

print("Predictions:", predict(X))
