import numpy as np
from decision_node import build_tree


class DecisionTree(object):
    """
    DecisionTree class, that represents one Decision Tree
    max_tree_depth: maximum depth for this tree.
    """
    def __init__(self, max_tree_depth):
        self.max_depth = max_tree_depth

    def fit(self, data):#X, Y):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array
        :param Y: 1 dimensional python list or numpy 1 dimensional array
        """
        self.trees = build_tree(data, current_depth=0, max_depth=self.max_depth)

    def predict(self, X):
        """
        :param X: 2 dimensional python list or numpy 2 dimensional array 
        return: Y - 1 dimension python list with labels
        """
        y = []
        for i in range(X.shape[0]):
            y.append(predict_for_each_row(self.trees, X[i]))
        return np.matrix(y).T

def predict_for_each_row(tree, row):
    if tree.is_leaf:
       for k, _ in tree.current_results.items():
            return k
    if row[tree.column] >= tree.value:
        return predict_for_each_row(tree.true_branch, row)
    else:
        return predict_for_each_row(tree.false_branch, row)