from prophets import *
from utils import *

K_LISTS = [2, 5, 10, 50]
M_LISTS = [1, 10, 50, 1000]
K_SCENARIO_3 = 500
EPSILON = 0.01
K_SCENARIO_6_C1 = 5
K_SCENARIO_6_C2 = 500

def run_experiment(lst_of_prophets, trainset_reduced, k):
    # Calculate each prophet estimation error on the k prophets
    for j in range(k):
        lst_of_prophets[j].est_error = compute_error(
            lst_of_prophets[j].predict(trainset_reduced), trainset_reduced)
    return lst_of_prophets


def check_answers(lst_of_prophets, test_set, count_best_prop, smaller_then_e,
                  est_avg_err, test_err, index_of_best_prophet ,train_gen ,test_gen ):

    chosen_prophet_index = np.argmin(
        [prophet.est_error for prophet in lst_of_prophets])

    # Check if the best prophet got chosen
    if chosen_prophet_index == index_of_best_prophet:
        count_best_prop += 1
        smaller_then_e += 1
    else:
        # check if the chosen prophet is one percent was the best prophet
        est_err = lst_of_prophets[chosen_prophet_index].get_err_prob() - lst_of_prophets[index_of_best_prophet].get_err_prob()
        est_avg_err += est_err

        if est_err < EPSILON:
            smaller_then_e += 1

    # Compute the selected prophet average error
    emprical_test_error = compute_error(
        lst_of_prophets[chosen_prophet_index].predict(test_set), test_set)

    test_err += emprical_test_error

    # Compute the selected prophet generalization error on train set
    train_gen += lst_of_prophets[chosen_prophet_index].get_err_prob() - lst_of_prophets[chosen_prophet_index].est_error

    # Compute the selected prophet generalization error on test set
    test_gen += abs( lst_of_prophets[chosen_prophet_index].get_err_prob() - emprical_test_error)



    return count_best_prop, smaller_then_e, est_avg_err, test_err , train_gen, test_gen


def Scenario_1(train_set, test_set):
    """
    Question 1.
    2 Prophets 1 Game.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    # The var that count thw wins of Prophet one
    p1_wins = 0

    #
    test_err = 0.

    #
    est_err = 0.

    # Starting Prophet one with his Probability
    p1 = Prophet(0.2)

    # Starting Prophet two with his Probability
    p2 = Prophet(0.4)

    # test
    for i in range(100):

        # chooses the one set the function is training on
        trainset_reduced = np.random.choice(train_set[i, :], size=1)

        # Predict prediction for profit one,and compute its error rate
        p1_err = compute_error(p1.predict(trainset_reduced), trainset_reduced)

        # Predict prediction for profit two,and compute its error rate
        p2_err = compute_error(p2.predict(trainset_reduced), trainset_reduced)

        # Checks witch profit has a better error rate
        if p1_err <= p2_err:

            p1_wins += 1

            test_err += compute_error(p1.predict(test_set), test_set)

        else:

            test_err += compute_error(p2.predict(test_set), test_set)

            est_err += 0.2

    print("Average test error of selected prophet: ", test_err / 100.)

    print("Number of times best prophet selected: ", p1_wins)

    print("Average approximation error: 0.2")

    print("Average estimation error: ", est_err / 100.)

    print()


def Scenario_2(train_set, test_set):
    """
    Question 2.
    2 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    # The var that count thw wins of Prophet one
    p1_wins = 0

    #
    test_err = 0.

    #
    est_err = 0.

    # Starting Prophet one with his error Probability
    p1 = Prophet(0.2)

    # Starting Prophet two with his error Probability
    p2 = Prophet(0.4)

    # test
    for i in range(100):

        # chooses the 10 sets the function is testing on
        trainset_reduced = np.random.choice(train_set[i, :], size=10)

        # Predict prediction for profit one,and compute its error rate
        p1.est_error = compute_error(p1.predict(trainset_reduced),
                                     trainset_reduced)

        # Predict prediction for profit two,and compute its error rate
        p2.est_error = compute_error(p2.predict(trainset_reduced),
                                     trainset_reduced)

        # Checks witch profit has a better error rate
        if p1.est_error <= p2.est_error:

            p1_wins += 1

            test_err += compute_error(p1.predict(test_set), test_set)

        else:

            test_err += compute_error(p2.predict(test_set), test_set)

            est_err += 0.2

    print("Average test error of selected prophet: ", test_err / 100.)

    print("Number of times best prophet selected: ", p1_wins)

    print("Average estimation error: ", est_err / 100.)

    print()


def Scenario_3(train_set, test_set):
    """
    Question 3.
    500 Prophets 10 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############

    count_best_prop = 0

    smaller_then_e = 0

    test_err = 0

    est_avg_err = 0.

    train_gen = 0
    test_gen =0



    # randomly draw 500 prophets with  error rate randomly between [0,1]
    lst_of_prophets = sample_prophets(K_SCENARIO_3, 0, 1)

    # gets the index of the best Prophet
    index_of_best_prophet = np.argmin(
        [prophet.get_err_prob() for prophet in lst_of_prophets])

    average_approx_error = lst_of_prophets[index_of_best_prophet].get_err_prob()

    # average_approx_error = np.mean(
    #     [prophet.get_err_prob() for prophet in lst_of_prophets])

    # Repeat experiment 100 times
    for i in range(100):
        # chooses the 10 sets that the Scenario is training on
        trainset_reduced = np.random.choice(train_set[i, :], size=10)

        # Calculate each prophet estimation error on the 500 games
        lst_of_prophets = run_experiment(lst_of_prophets, trainset_reduced,
                                         K_SCENARIO_3)

        count_best_prop, smaller_then_e, est_avg_err, test_err , train_gen , test_gen= check_answers(
            lst_of_prophets, test_set, count_best_prop, smaller_then_e,
            est_avg_err, test_err, index_of_best_prophet , train_gen , test_gen)

    print("Average test error of selected prophet: ", test_err / 100)

    print("Number of times best prophet selected: ", count_best_prop)

    print("Number of times good enough prophet selected: ", smaller_then_e)

    print("Average approximation error: ", average_approx_error)

    print("Average estimation error: ", est_avg_err / 100.)

    print()


def Scenario_4(train_set, test_set):
    """
    Question 4.
    500 Prophets 1000 Games.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    # randomly draw 500 prophets with a error rate ramdomly
    lst_of_prophets = sample_prophets(K_SCENARIO_3, 0, 1)

    # gets the index of the best Prophet
    index_of_best_prophet = np.argmin(
        [prophet.get_err_prob() for prophet in lst_of_prophets])

    average_approx_error = lst_of_prophets[index_of_best_prophet].get_err_prob()

    count_best_prop = 0

    smaller_then_e = 0

    test_err = 0

    est_avg_err = 0.

    train_gen = 0

    test_gen =0

    # Repeat experiment 100 times
    for i in range(100):
        # chooses the 1000 sets the function is testing on
        trainset_reduced = np.random.choice(train_set[i, :], size=1000)

        # Calculate each prophet estimation error on the 500 games
        lst_of_prophets = run_experiment(lst_of_prophets, trainset_reduced,
                                         K_SCENARIO_3)

        count_best_prop, smaller_then_e, est_avg_err, test_err , train_gen , test_gen= check_answers(
            lst_of_prophets, test_set, count_best_prop, smaller_then_e,
            est_avg_err, test_err, index_of_best_prophet , train_gen , test_gen)
            
            


    print("Average test error of selected prophet: ", test_err / 100.)

    print("Number of times best prophet selected: ", count_best_prop)

    print("Number of times good enough prophet selected: ", smaller_then_e)

    print("Average approximation error:" ,average_approx_error )

    print("Average estimation error: ", est_avg_err / 100.)

    print("Average generalization error on train set =: ", train_gen / 100.)

    print("Average generalization error on test set =: ", test_gen / 100.)
    print()

def Scenario_5(train_set, test_set):
    """
    Question 5.
    School of Prophets.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############


    data= np.zeros(shape=(len(K_LISTS), len(M_LISTS)), dtype=object)

    columns = [f"m={m}" for m in M_LISTS]
    index =  [f"k={k}" for k in K_LISTS]


    # loop on the number k options that will represent the amount of prophets
    for k in K_LISTS:

        lst_of_prophets = sample_prophets(k, 0, 0.2)

        # gets the index of the best Prophet
        index_of_best_prophet = np.argmin(
            [prophet.get_err_prob() for prophet in lst_of_prophets])

        average_approx_error = lst_of_prophets[index_of_best_prophet].get_err_prob()
        #average_approx_error = np.mean([prophet.get_err_prob() for prophet in lst_of_prophets])

        # loop on the  number M options that will represent the amount of sets
        for m in M_LISTS:

            avg_est_err = 0.

            # Repeat experiment 100 times
            for i in range(100):
                trainset_reduced = np.random.choice(train_set[i, :], size=m)

                lst_of_prophets = run_experiment(lst_of_prophets,
                                                 trainset_reduced,
                                                 k)

                chosen_prophet_index = np.argmin([prophet.est_error for prophet in lst_of_prophets])
                avg_est_err += lst_of_prophets[chosen_prophet_index].get_err_prob() - lst_of_prophets[index_of_best_prophet].get_err_prob()

            data[K_LISTS.index(k)][M_LISTS.index(m)] = f"(Est: {(round( avg_est_err / 100., 4)):}, Approx: {round(average_approx_error, 4):})"

    pd.set_option('colheader_justify', 'center')

    df = pd.DataFrame(data=data, index=index, columns=columns)
    print(df.to_string())
    print()

def Scenario_6(train_set , test_set ):
    """
    Question 6.
    The Bias-Variance Tradeoff.
    You may change the input & output parameters of the function as you wish.
    """
    ############### YOUR CODE GOES HERE ###############
    lst_of_prophets_1 = sample_prophets(K_SCENARIO_6_C1, 0.3, 0.6)
    lst_of_prophets_2 = sample_prophets(K_SCENARIO_6_C2, 0.25, 0.26)

    # gets the index of the best Prophet
    index_of_best_prophet_1 = np.argmin([prophet.get_err_prob() for prophet in lst_of_prophets_1])
    index_of_best_prophet_2 = np.argmin([prophet.get_err_prob() for prophet in lst_of_prophets_2])

    average_approx_error_1 = lst_of_prophets_1[index_of_best_prophet_1].get_err_prob()
    average_approx_error_2 = lst_of_prophets_2[index_of_best_prophet_2].get_err_prob()

    # average_approx_error_1 = np.mean([prophet.get_err_prob() for prophet in lst_of_prophets_1])
    # average_approx_error_2 = np.mean([prophet.get_err_prob() for prophet in lst_of_prophets_2])

    test_err_1 = 0
    test_err_2 = 0

    est_avg_err_1 = 0.
    est_avg_err_2 = 0.

    # Repeat experiment 100 times
    for i in range(100):

        # chooses the 1000 sets the function is testing on
        trainset_reduced = np.random.choice(train_set[i, :], size=1000)

        # Calculate each class and each prophet estimation error on the 500 games
        lst_of_prophets_1 = run_experiment(lst_of_prophets_1, trainset_reduced,K_SCENARIO_6_C1)
        lst_of_prophets_2 = run_experiment(lst_of_prophets_2, trainset_reduced,K_SCENARIO_6_C2)

        chosen_prophet_1_index = np.argmin([prophet.est_error for prophet in lst_of_prophets_1])
        chosen_prophet_2_index = np.argmin([prophet.est_error for prophet in lst_of_prophets_2])

        est_avg_err_1 += lst_of_prophets_1[chosen_prophet_1_index].get_err_prob() - lst_of_prophets_1[index_of_best_prophet_1].get_err_prob()
        est_avg_err_2 += lst_of_prophets_2[chosen_prophet_2_index].get_err_prob() - lst_of_prophets_2[index_of_best_prophet_2].get_err_prob()

        test_err_1 += compute_error(lst_of_prophets_1[chosen_prophet_1_index].predict(test_set), test_set)
        test_err_2 += compute_error(lst_of_prophets_2[chosen_prophet_2_index].predict(test_set), test_set)

    print("Hypothesis Class 1:")
    print("Approximation Error:", average_approx_error_1)
    print("Average Estimation Error:", est_avg_err_1)
    print("Prediction error:", test_err_1 )
    print()
    print("Hypothesis Class 2:")
    print("Average Approximation Error:", average_approx_error_2)
    print("Average Estimation Error:", est_avg_err_2)
    print("Prediction error:" , test_err_2 )


if __name__ == '__main__':
    np.random.seed(0)  # DO NOT MOVE / REMOVE THIS CODE LINE!

    # train, validation and test splits for Scenario 1-3, 5
    train_set = create_data(100, 1000)
    test_set = create_data(1, 1000)[0]

    print(f'Scenario 1 Results:')
    Scenario_1(train_set, test_set)

    print(f'Scenario 2 Results:')
    Scenario_2(train_set, test_set)

    print(f'Scenario 3 Results:')
    Scenario_3(train_set, test_set)

    print(f'Scenario 4 Results:')
    Scenario_4(train_set, test_set)

    print(f'Scenario 5 Results:')
    Scenario_5(train_set, test_set)

    print(f'Scenario 6 Results:')
    Scenario_6(train_set, test_set)
