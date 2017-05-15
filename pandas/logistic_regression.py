import numpy as np

def sigmoid(s):
    theta = 1 / (1. + np.exp(-s))
    return theta

def gradient_descent(X, Y, epsilon=1e-8, step_size=1e-4, max_steps=1000):
    """
    Implement gradient descent using full value of the gradient.
    :param X: data matrix (2 dimensional np.array)
    :param Y: response variables (1 dimensional np.array)
    :param l: regularization parameter lambda
    :param epsilon: approximation strength
    :param max_steps: maximum number of iterations before algorithm will terminate.
    :return: value of beta (1 dimensional np.array)
    """
    beta = np.zeros(X.shape[1])
    beta = beta.reshape((-1, 1))
    for s in range(max_steps):
        ngrad = normalized_gradient(X, Y, beta)
        beta = beta + step_size * ngrad
        if np.linalg.norm(step_size * ngrad) < epsilon * np.linalg.norm(beta):
            break
    return beta

def feature_normalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(1, X.shape[1]):
        X[:,i] = (X[:,i] - mu[i]*np.ones(X.shape[0]) )/(sigma[i])
    return X

def normalized_gradient(X, Y, beta):
    X = feature_normalize(X)
    N = X.shape[0]
    norm_grad = (X.T.dot(sigmoid(X.dot(beta)) - Y)) / N
    return norm_grad