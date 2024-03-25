P = 0.5
HOME = 1
AWAY = 0

import pandas as pd
import numpy as np


# Note: You are not allowed to add any additional imports!

def create_data(n_sets, n_samples):
    """
    Creates a 2-d numpy array of labels.
    y values are randomly selected from {0, 1}
    :param n_sets: number of sets
    :param n_samples: number of points
    :return: y
    """
    ############### YOUR CODE GOES HERE ###############

    # creates a 2_d array
    y = np.random.choice([HOME, AWAY], p=[P, P], size=(n_sets, n_samples))

    # return the 2-D array
    return y


# def compute_error(preds, gt):
#     """
#     Computes the error of the predictions
#     :param preds: predictions
#     :param gt: ground truth
#     :return: error
#     """
#     ############### YOUR CODE GOES HERE ###############
#
#     # The amount of games predicts
#     games = len(gt)
#
#     # counting the amount of games that were predicted correctly
#     correct = np.count_nonzero(gt == preds)
#
#     error_rate = 1.0 - (correct / games)
#
#     # returning the average error score
#     return error_rate


def compute_error(preds, gt):
    """
    Computes the error of the predictions
    :param preds: predictions
    :param gt: ground truth
    :return: error
    """
    ############### YOUR CODE GOES HERE ###############
    return np.average(np.abs(preds-gt))

