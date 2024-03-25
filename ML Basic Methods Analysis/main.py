from knn import KNNClassifier
from tree import *

K = [1, 10, 100, 1000, 3000]
L = ["l1", "l2"]
N = 50

MAX_IND2 = 0
MAX_IND1 = 5
MiN_IND = 9

CASE_1 = "Decision Boundaries: Kmax , l2"
CASE_2 = "Decision Boundaries: Kmim , l2"
CASE_3 = "Decision Boundaries: Kmax , l1"

MAXIMAL_DEPTH = [1, 2, 4, 6, 10, 20, 50, 100]
MAXIMAL_LEAF_NODES = [50, 100, 1000]

TRAIN = 0
VALIDATION = 1
TEST = 2

CASE_4 = " Best Tree Visualization on the Validation Set:"
CASE_5 = " Best Tree Visualization on the Validation set with only 50 Node:"
CASE_6 = " Best Tree Visualization on the Validation set with up to 6 depth"
CASE_7 = " Forest  "
CASE_8 = " xgboost  "


def chart_data(row_headers, column_headers, data):
    df = pd.DataFrame(data, index=row_headers, columns=column_headers)
    return df


def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.

    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh

    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)

    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200

    # Plot the decision boundary.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))

    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])

    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)

    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40,
                alpha=0.7, vmin=vmin, vmax=vmax)

    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def show_anomaly(train, anomaly, normal):
    # colored in black and with an opacity of 0.01
    plt.scatter(train[:, 0], train[:, 1], color='black', alpha=0.01)

    # color the normal points in blue
    plt.scatter(normal[:, 0], normal[:, 1], color="blue", label="normal")

    # color anomalous points in red
    plt.scatter(anomaly[:, 0], anomaly[:, 1], color="red", label="Anomalous")

    # Label the chart
    plt.title("Anomaly Detection")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.show()


def highest_N_anomaly_scores(AD_test, sorted_sum_ind, N):
    # the 50 test examples with the highest anomaly scores
    anomaly = AD_test[sorted_sum_ind[-N:], :]

    # the rest of the points are defined as normal
    normal = AD_test[sorted_sum_ind[:-N], :]

    return anomaly, normal


def read_data(filename):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    """

    # the data in pandas dataframe format
    df = pd.read_csv(filename)

    # extract the column names
    col_names = list(df.columns)

    # the data in numpy array format
    features = df[[col_names[0], col_names[1]]].values

    labels = None

    if (len(col_names) == 3):
        labels = df[col_names[2]].values

    return features, labels


def knn_algo(X_train, Y_train, X_test, Y_test, k, distance_metric):
    """
    Notice the similarity to the decision tree demo above.
    This is the sklearn standard format for models.
    """

    # Initialize the KNNClassifier with k and the distance metric
    knn_classifier = KNNClassifier(k, distance_metric)

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # Predict the labels for the test set
    y_pred = knn_classifier.predict(X_test)

    # Calculate the accuracy of the classifier
    accuracy = np.mean(y_pred == Y_test)

    knn_classifier.set_accuracy(accuracy)

    return knn_classifier, accuracy


def Q5_1_1(X_train, Y_train, X_test, Y_test):
    # the user should allow the user to specify the number k
    k = int(input("Please enter K: "))

    results = np.array([])

    knn_1, accuracy_1 = knn_algo(X_train, Y_train, X_test, Y_test, k, L[0])
    knn_2, accuracy_2 = knn_algo(X_train, Y_train, X_test, Y_test, k, L[1])

    # add data to results
    results = np.append(results, accuracy_1)
    results = np.append(results, accuracy_2)

    # Reshape the array to (2, 5)
    results = np.column_stack(results).reshape(2, 1)
    df = chart_data(L, [k], results)
    print(df)


def Q5_1_2(X_train, Y_train, X_test, Y_test):
    # Save all 10 models
    models = np.array([])

    results = np.array([])

    # For every distance_metric
    for dis in L:

        # For every distance_metric
        for k in K:
            # Runs the KNN algo for each (L , k)
            knn, accuracy = knn_algo(X_train, Y_train, X_test, Y_test, k, dis)

            # add results to np array
            results = np.append(results, accuracy)

            # add object to np array
            models = np.append(models, knn)

    # Reshape the array to (2, 5)
    results = np.column_stack(results).reshape(2, 5)

    # Enter data to a chart
    df = chart_data(L, K, results)

    print(df)
    return models


def Q5_2_2(models, X_test, Y_test):
    # the k model with the highest test accuracy when using the L2 distance metric
    kmax_model = models[MAX_IND1]

    #  Choose the k model with the lowest test accuracy when using the L2 distance metric
    kmin_model = models[MiN_IND]

    #  Choose the k model with the highest test accuracy when using the L1 distance metric
    kmax_l1_model = models[MAX_IND2]

    plot_decision_boundaries(kmax_model, X_test, Y_test, CASE_1)

    plot_decision_boundaries(kmin_model, X_test, Y_test, CASE_2)

    plot_decision_boundaries(kmax_l1_model, X_test, Y_test, CASE_3)


def Q5_3_2(X_train, Y_train, AD_test):
    # Initialize the KNNClassifier with k and the distance metric
    knn_classifier = KNNClassifier(5, L[1])

    # Train the classifier
    knn_classifier.fit(X_train, Y_train)

    # for sample in AD_test:
    knn_distances, knn_ind_label = knn_classifier.knn_distance(AD_test)

    # sum the 5 knn_distances for each Feature
    distances_sum = np.sum(knn_distances, axis=1).flatten()

    # sorts the sum knn_distances
    distances_sorted_sum_index = np.argsort(distances_sum)

    return knn_distances, distances_sorted_sum_index


def Q5_3_4(X_train, AD_test, sorted_sum_index):
    # splitting up the AD_teat by N the highest anomaly scores
    anomaly, normal = highest_N_anomaly_scores(AD_test, sorted_sum_index, N)

    show_anomaly(X_train, anomaly, normal)


def Q6_1_data(X_train, Y_train, X_test, Y_test, X_validation, Y_validation):
    models = np.array([])

    for max_dep in MAXIMAL_DEPTH:

        for max_leaf in MAXIMAL_LEAF_NODES:
            tree = Tree(max_dep, max_leaf)

            # fit the tree
            tree.fit(X_train, Y_train)

            # prodict and save accuracy for each set
            tree.accuracy(X_train, Y_train, TRAIN)

            tree.accuracy(X_validation, Y_validation, VALIDATION)

            tree.accuracy(X_test, Y_test, TEST)

            # add object to np array
            models = np.append(models, tree)

    return models


def sort_tree(models):
    # Get the indices that would sort the array based on accuracy_validation
    sorted_indices = np.argsort([model.accuracy_validation for model in models])

    # Rearrange the models array based on the sorted indices
    sorted_models = models[sorted_indices]

    return sorted_models


def print_trees_test_accuracies(sorted_models):
    i = 1
    for mod in sorted_models:
        print(f"tree numer {i} - training accuracy: ", mod.accuracy_validation)
        i += 1


def Q6_1_qustions(models, X_validation, Y_validation):
    # Get a sorted  array based on accuracy_validation
    sorted_models = sort_tree(models)

    print("the tree :", sorted_models[-1])
    print("the tree with the best validation accuracy:",
          sorted_models[-1].accuracy_validation)

    print("same trees training accuracy:", sorted_models[-1].accuracy_training)
    print("same trees test accuracy:", sorted_models[-1].accuracy_test)
    print("")

    print_trees_test_accuracies(sorted_models)

    plot_decision_boundaries(sorted_models[-1], X_validation, Y_validation, CASE_4)

    # Filter models with max_leaf_nodes == 50
    filtered_models = [model for model in sorted_models if model.max_leaf_nodes == 50]
    plot_decision_boundaries(filtered_models[-1], X_validation, Y_validation, CASE_5)

    # Filter models with max_depth <= 6
    filtered_models_2 = [model for model in sorted_models if model.max_depth <= 6]
    plot_decision_boundaries(filtered_models_2[-1], X_validation, Y_validation, CASE_6)


def Q6_7(X_train, Y_train, X_test, Y_test, X_validation, Y_validation):
    forest = loading_random_forest(X_train, Y_train)
    y_pred = forest.predict(X_validation)

    forest_accuracy = np.mean(y_pred == Y_validation)

    print("Forest accuracy: ", forest_accuracy)

    plot_decision_boundaries(forest, X_validation, Y_validation, CASE_7)


def bounos(X_train, Y_train, X_test, Y_test, X_validation, Y_validation):
    #
    xgboost = loading_xgboost(X_train, Y_train)

    #
    y_pred = xgboost.predict(X_validation)
    vel_acc = np.mean(y_pred == Y_validation)

    #
    y_pred = xgboost.predict(X_test)
    test_acc = np.mean(y_pred == Y_test)

    #
    print("xgboost test accuracy ", test_acc)
    plot_decision_boundaries(xgboost, X_test, Y_test, CASE_8)


if __name__ == '__main__':
    ### Classification with k-Nearest Neighbors (kNN) ###
    # Load the training files
    X_train, Y_train = read_data("train.csv")

    # Load the test files
    X_test, Y_test = read_data("test.csv")

    # Run Question 5.1.1
    Q5_1_1(X_train, Y_train, X_test, Y_test)

    # Run Question 5.1.2
    models = Q5_1_2(X_train, Y_train, X_test, Y_test)

    # Run Question 5.2.2
    Q5_2_2(models, X_test, Y_test)

    ### Anomaly Detection Using kNN ###
    # Load the AD_test files
    AD_test, nothing = read_data("AD_test.csv")

    # Run Question 5.3
    knn_distances, distances_sorted_sum_index = Q5_3_2(X_train, Y_train, AD_test)
    Q5_3_4(X_train, AD_test, distances_sorted_sum_index)

    ### Decision Trees ###
    # Load the validation files
    X_validation, Y_validation = read_data("validation.csv")

    # Run Question 6
    models_2 = Q6_1_data(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)
    Q6_1_qustions(models_2, X_validation, Y_validation)

    Q6_7(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)

    ### bouns ###
    bounos(X_train, Y_train, X_test, Y_test, X_validation, Y_validation)
