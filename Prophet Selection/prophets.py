HOME = 1
AWAY = 0

import numpy as np


def sample_prophets(k, min_p, max_p):
    """
    Samples a set of k prophets
    :param k: number of prophets
    :param min_p: minimum probability
    :param max_p: maximum probability
    :return: list of prophets
    """
    # Start a list
    lst_of_prophets = []

    for i in range(k):
        # Add to list
        lst_of_prophets.append(
            Prophet(np.random.uniform(low=min_p, high=max_p)))

    # return the list of prophets
    return lst_of_prophets


class Prophet:

    def __init__(self, err_prob):
        """
        Initializes the Prophet model
        :param err_prob: the probability of the prophet to be wrong
        """
        ############### YOUR CODE GOES HERE ###############
        self.___err_prob = err_prob
        self.est_error = 0

    def __str__(self):
        p_str = ""
        err_p = str(self.___err_prob)
        p_str += err_p + "\n"
        return p_str

    def predict(self, y):
        """
        Predicts the label of the input point
        draws a random number between 0 and 1
        if the number is less than the probability, the prediction is correct (according to y)
        else the prediction is wrong
        NOTE: Realistically, the prophet should be a function from x to y (without getting y as an input)
        However, for the simplicity of our simulation, we will give the prophet y straight away
        :param y: the true label of the input point
        :return: a prediction for the label of the input point
        """
        ############### YOUR CODE GOES HERE ###############

        # draw a random number between [0,1]
        t = np.random.uniform(size=y.shape)

        # for each sec in the array, Check if the random number is less
        # than the probability number,and place T/F in th array
        correct_pred = t < (1.0 -self.___err_prob)

        # Create a copy of y to avoid modifying the original array
        prediction = y.copy()

        # Keep correct predictions as they are
        prediction[correct_pred] = y[correct_pred]

        # Handle incorrect predictions by checking the value of y
        # and placing the other value
        prediction[~correct_pred] = np.where(y[~correct_pred] == HOME, AWAY,
                                             HOME)

        return prediction

    def get_err_prob(self):
        return self.___err_prob