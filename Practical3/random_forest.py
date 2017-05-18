from decision_node import build_tree
from decision_tree import predict_for_each_row
import numpy as np


class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """

    def __init__(self, num_trees, max_tree_depth, ratio_per_tree):
        self.num_trees = num_trees
        self.ratio_per_tree = ratio_per_tree
        self.max_tree_depth = max_tree_depth
        self.trees = None

    def fit(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        # with randomized data.
        self.trees = []
        part = int(self.ratio_per_tree * len(X))
        for i in range(self.num_trees):
            np.random.shuffle(X)
            self.trees.append(build_tree(X[0:part], 0, self.max_tree_depth))

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :return: (Y, conf), tuple with Y being 1 dimension python
        list with labels, and conf being 1 dimensional list with
        confidences for each of the labels.
        """
        # TODO: Evaluate labels in each of the `self.tree`s and return the
        # label and confidence with the most votes for each of
        # the data points in `X`
        X = np.array(X)
        Y = []
        for i in range(len(X)):
            y=[]
            for tree in self.trees:
                y.append(predict_for_each_row(tree, X[i]))
            best_value = max(set(y), key = y.count)
            Y.append(best_value)
        return np.matrix(Y).T