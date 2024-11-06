import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

# Task 1

learning_rate = 0.05

# Data collection
df = pd.read_csv("MNIST_CV.csv")

# Data preprocessing
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

kf = KFold(n_splits=10)

for i, (train_index, test_index) in enumerate(kf.split(X)):
    # print(f"Fold {i}")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")

    logistic = Sigmoid(X_training.dot(coefficient.T))
    costp = (-logistic + np.squeeze(y_training).T.dot(X_training)) ##gradient calc (direction)
    y_training = np.squeeze(y_training)
    coefficient = coefficient + learning_rate * costp

    # likelihood function y.log(p(xi) - ((1-y)(log(1-p(xi)))))
    lw1 = (y_training * np.log(logistic))

    lw2 = ((1-y_training) * np.log(1-logistic))
    costv = +lw1 + lw2

    # mean of function
    costf = np.mean(costv)
    cost.append(costf)