import faiss as faiss
import numpy as np


class KNNClassifier:
    def __init__(self, k, distance_metric='l2'):
        self.k = k

        self.distance_metric = distance_metric

        # feature vectors
        self.X_train = None

        # labels
        self.Y_train = None

        # test accuracy
        self.test_accuracy = 0

    def __str__(self):
        if self.test_accuracy == 0:
            return f"KNN(k={self.k}, distance_metric='{self.distance_metric}')"

        return f"KNN(k={self.k}, distance_metric='{self.distance_metric}' , accuracy ={self.test_accuracy})"

    def fit(self, X_train, Y_train):
        """
        Update the kNN classifier with the provided training data.

        Parameters:
        - X_train (numpy array) of size (N, d): Training feature vectors.
        - Y_train (numpy array) of size (N,): Corresponding class labels.
        """
        self.X_train = X_train.astype(np.float32)
        self.Y_train = Y_train
        d = self.X_train.shape[1]
        if self.distance_metric == 'l2':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L2)
        elif self.distance_metric == 'l1':
            self.index = faiss.index_factory(d, "Flat", faiss.METRIC_L1)
        else:
            raise NotImplementedError
        pass
        self.index.add(self.X_train)

    def predict(self, X):
        """
        Predict the class labels for the given data.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M,): Predicted class labels.
        """
        # Calculate kNN distances and indices
        knn_distances, knn_ind_label = self.knn_distance(X)

        # Get labels of the K nearest neighbors
        nearest_labels = self.Y_train[knn_ind_label]

        # Initialize an array to store predicted labels
        predictions = np.empty((X.shape[0],), dtype=self.Y_train.dtype)

        # For each test instance, find the majority class among k nearest neighbors
        for i in range(len(nearest_labels)):
            # get nearest labels from training data and count
            unique_labels, label_counts = np.unique(nearest_labels[i], return_counts=True)

            frequently_label = unique_labels[np.argmax(label_counts)]

            predictions[i] = frequently_label

        # Predicted class labels
        return predictions.flatten()

    def knn_distance(self, X):
        """
        Calculate kNN distances for the given data. You must use the faiss library to compute the distances.
        See lecture slides and https://github.com/facebookresearch/faiss/wiki/Getting-started#in-python-2 for more information.

        Parameters:
        - X (numpy array) of size (M, d): Feature vectors.

        Returns:
        - (numpy array) of size (M, k): kNN distances.
        - (numpy array) of size (M, k): Indices of kNNs.
        """
        X = X.astype(np.float32)
        return self.index.search(X, self.k)

    def set_accuracy(self, acc):
        self.test_accuracy = acc
