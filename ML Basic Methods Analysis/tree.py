import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

TRAIN = 0
VALIDATION = 1
TEST = 2
class Tree:

    def __init__(self ,max_depth , max_leaf_nodes) :
        #
        self.tree_classifier = DecisionTreeClassifier(max_depth=max_depth , max_leaf_nodes = max_leaf_nodes)
        #
        self.max_depth = max_depth
        #
        self.max_leaf_nodes = max_leaf_nodes
        #
        self.accuracy_training = 0

        self.accuracy_validation = 0

        self.accuracy_test = 0

    def __str__(self):
        return f"Tree(max_depth={self.max_depth}, max_leaf_nodes={self.max_leaf_nodes}, " \
               f"accuracy_training={self.accuracy_training}, accuracy_validation={self.accuracy_validation}, " \
               f"accuracy_test={self.accuracy_test})"

    def fit(self, X_train, Y_train):
        # Train the Decision Tree on the training data
        self.tree_classifier.fit(X_train, Y_train)


    def predict(self, X_test):
        # Make predictions on the test data
        y_pred = self.tree_classifier.predict(X_test)

        return y_pred

    def accuracy(self, X_test , Y_test , i):

        # Make predictions on the test data
        y_pred = self.tree_classifier.predict(X_test)

        # Compute the accuracy of the predictions
        accuracy = np.mean(y_pred == Y_test)

        if i == TRAIN:
            self.accuracy_training = accuracy

        elif i == VALIDATION:
            self.accuracy_validation = accuracy

        else:
            self.accuracy_test = accuracy


def loading_random_forest(X_train , Y_train):
    model = RandomForestClassifier(n_estimators=300, max_depth=6, n_jobs=4)
    model.fit(X_train , Y_train)
    return model

def loading_xgboost(X_train , Y_train):
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=4)
    model.fit(X_train , Y_train)
    return model
