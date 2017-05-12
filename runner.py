import numpy as np
import matplotlib.pyplot as plt
import random

from decision_tree import DecisionTree
from random_forest import RandomForest
from logistic_regression import gradient_descent
from logistic_regression import sigmoid

def accuracy_score(Y_true, Y_predict):
    accuracy = 0
    for i in range(len(Y_true)):
        if Y_true[i] == Y_predict[i]:
            accuracy = accuracy + 1
    return accuracy / len(Y_true) * 100.0

def evaluate_performance():
    '''
    Evaluate the performance of decision trees and logistic regression,
    average over 1,000 trials of 10-fold cross validation

    Return:
      a matrix giving the performance that will contain the following entries:
      stats[0,0] = mean accuracy of decision tree
      stats[0,1] = std deviation of decision tree accuracy
      stats[1,0] = mean accuracy of logistic regression
      stats[1,1] = std deviation of logistic regression accuracy

    ** Note that your implementation must follow this API**
    '''

    # Load Data
    filename = 'SPECTF.dat'
    data = np.loadtxt(filename, delimiter=',')
    X = data[:, 1:]
    y = np.array([data[:, 0]]).T
    #print('dim of X =', X.shape)
    #print('dim of y = ', y.shape)
    n, d = X.shape

    all_accuracies_tree = []
    all_accuracies_forest = []
    all_accuracies_log_reg = []

    NUMBER_OF_TREES = 10
    MAX_TREE_DEPTH_RF = 100
    MAX_TREE_DEPTH_TR = 100
    RATIO_PER_TREE = 0.7

    for trial in range(1):
        print('trial', trial + 1)
        idx = np.arange(n)
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # divide data into train-test
        folds, i = 10, 3
        Xtest = X[i::folds]  # i::folds = i-th element, (i+folds)th element, ...
        ytest = y[i::folds]
        Xtrain = np.array([X[j] for j in range(len(X)) if (j % folds) != i])
        ytrain = np.array([y[j] for j in range(len(y)) if (j % folds) != i])
        train_data = np.column_stack((Xtrain, ytrain))
        #idx = list(range(n))
        #random.shuffle(idx)

        #print('dim of ytrain', ytrain.shape) # (240,1)
        #print('dim of Xtrain', Xtrain.shape) # (240,44)
        #print('dim of ytest', ytest.shape) # (27,1)
        #print('dim of Xtest', Xtest.shape) # (27,44)

        # train the decision tree
        #classifier_tree = DecisionTree(MAX_TREE_DEPTH_TR)
        #classifier_tree.fit(train_data)
        #y_pred = classifier_tree.predict(Xtest)
        #print('dim of y_pred', y_pred.shape)
        #accuracy_tree = accuracy_score(ytest, np.array(y_pred))
        #all_accuracies_tree.append(accuracy_tree)

        # train the random forest
        #classifier_forest = RandomForest(NUMBER_OF_TREES, MAX_TREE_DEPTH_RF, RATIO_PER_TREE)
        #classifier_forest.fit(train_data)
        #y_pred_forest = classifier_forest.predict(Xtest)
        #print('dim of y_pred_forest', y_pred_forest.shape)
        #accuracy_forest = accuracy_score(ytest, np.array(y_pred_forest))
        #all_accuracies_forest.append(accuracy_forest)
        #print('forest has been classified')

        # train by logistic regression
        beta = gradient_descent(np.column_stack((np.ones(len(Xtrain)),Xtrain)), ytrain, max_steps=5)
        y_pred_log_reg = []
        for i in range(Xtest.shape[0]):
            sigm = sigmoid((np.column_stack((np.ones(len(Xtest)), Xtest))).dot(beta))
            if sigm[i] >= 0.5:
                y_pred_log_reg.append(1)
            else:
                y_pred_log_reg.append(0)
        accuracy_log_reg = accuracy_score(ytest, np.array(y_pred_log_reg))
        all_accuracies_log_reg.append(accuracy_log_reg)

        #print("\nytest")
        #print(ytest)
        #print("\ny_pred tree")
        #print(y_pred)
        #print("\ny_pred forest")
        #print(y_pred_forest)
        
    # compute the training accuracies of the model
    #meanDecisionTreeAccuracy = np.mean(all_accuracies_tree)
    #stddevDecisionTreeAccuracy = np.std(all_accuracies_tree)

    #meanRandomForestAccuracy = np.mean(all_accuracies_forest)
    #stddevRandomForestAccuracy = np.std(all_accuracies_forest)

    meanLogisticRegressionAccuracy = np.mean(all_accuracies_log_reg)
    stddevLogisticRegressionAccuracy = np.std(all_accuracies_log_reg)

    # make certain that the return value matches the API specification
    stats = np.zeros((3, 2))
    stats[0, 0] = 0#meanDecisionTreeAccuracy
    stats[0, 1] = 0#stddevDecisionTreeAccuracy
    stats[1, 0] = 0#meanRandomForestAccuracy
    stats[1, 1] = 0#stddevRandomForestAccuracy
    stats[2, 0] = meanLogisticRegressionAccuracy
    stats[2, 1] = stddevLogisticRegressionAccuracy
    return stats

if __name__ == "__main__":
    stats = evaluate_performance()
    print("Decision Tree Accuracy = ", stats[0, 0], " (", stats[0, 1], ")")
    print("Random Forest Tree Accuracy = ", stats[1, 0], " (", stats[1, 1], ")")
    print("Logistic Reg. Accuracy = ", stats[2, 0], " (", stats[2, 1], ")")