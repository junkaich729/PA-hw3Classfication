import numpy as np
from collections import Counter
from scipy.spatial import distance

class MyKNN:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = []
        for row in X:
            label = self._predict(row)
            predictions.append(label)
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [distance.euclidean(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.n_neighbors]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# # Now you can use MyKNN just like you would use KNeighborsClassifier from scikit-learn
# knn = MyKNN(n_neighbors=3)
# knn.fit(X_train_pca, y_train)
# y_pred = knn.predict(X_test_pca)

# # Evaluate the custom KNN model
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Custom KNN Accuracy: {accuracy:.2f}")
