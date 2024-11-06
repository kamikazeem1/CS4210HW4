import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Task 1

# Data collection
df = pd.read_csv("MNIST_CV.csv")

# Data preprocessing
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def logistic_regression(X_training, y_training, learning_rate = 0.01, iterations=1000):
    m, n = X.shape # m-rows, n-columns
    coefficient = np.zeros(n) # init coefficient array
    for i in range(iterations):
        logistic = Sigmoid(X_training.dot(coefficient.T))
        costp = (-logistic + np.squeeze(y_training).T.dot(X_training)) ##gradient calc (direction)
        y_training = np.squeeze(y_training)
        coefficient = coefficient + learning_rate * costp
    return coefficient

def k_fold_cross_validation(X, y, k, learning_rate = 0.01, iterations=1000):
    kf = KFold(n_splits=k)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {i}")
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        coefficient = logistic_regression(X_train, y_train, learning_rate, iterations)


        

# likelihood function y.log(p(xi) - ((1-y)(log(1-p(xi)))))
lw1 = (y_training * np.log(logistic))

lw2 = ((1-y_training) * np.log(1-logistic))
costv = +lw1 + lw2

# mean of function
costf = np.mean(costv)
cost.append(costf)