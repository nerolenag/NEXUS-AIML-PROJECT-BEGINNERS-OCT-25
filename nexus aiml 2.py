import numpy as np
import pandas as pd
from collections import Counter

data = {
    'X1': [1, 2, 3, 6, 7, 8],
    'X2': [1, 2, 1, 6, 7, 8],
    'label': [0, 0, 0, 1, 1, 1]
}
df = pd.DataFrame(data)

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def knn_predict(X_train, y_train, x_test, k=3):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))
    
    distances.sort(key=lambda x: x[0])
    k_neighbors = [label for (_, label) in distances[:k]]
   
    return Counter(k_neighbors).most_common(1)[0][0]

X_train = df[['X1', 'X2']].values
y_train = df['label'].values
x_test = np.array([5, 5])

pred = knn_predict(X_train, y_train, x_test, k=3)
print(f"Predicted class for {x_test}: {pred}")
