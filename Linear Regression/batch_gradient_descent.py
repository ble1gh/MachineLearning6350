#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:50:49 2024

@author: brookleigh
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#learning rate
r = 0.1

#error threshold
epsilon = 1e-6

features = ['Cement', 'Slag', 'Fly Ash', 'Water','SP', 'Coarse Aggregate', 'Fine Aggregate', 'label']
features_test = ['Cement', 'Slag', 'Fly Ash', 'Water','SP', 'Coarse Aggregate', 'Fine Aggregate']

#read in data
df = pd.read_csv('concrete/train.csv', names=features)
dftest = pd.read_csv('concrete/test.csv', names=features)


def batch_gradient_descent(df, features, learning_rate=0.01, epsilon=1e-6):
    addones = np.ones(len(df))
    df.insert(0, 'ones', addones, allow_duplicates=True)
    x = df.to_numpy()

    # Initialize w vector
    w = np.zeros(len(features))

    norm_diff = 1
    count = 0
    cost_values = []

    while abs(norm_diff) > epsilon:
        grad = np.zeros(len(w))
        cost = 0

        for i in range(len(df)):
            xi = x[i, :-1]
            yi = x[i, -1]
            e = yi - np.dot(xi, w)
            cost += 0.5 * e ** 2
            grad += -e * xi

        w_previous = w.copy()
        w -= grad * learning_rate
        norm_diff = np.linalg.norm(w - w_previous)
        cost_values.append(cost)
        count += 1

    return w, cost_values, count

def calculate_error(w, dftest):
    addones = np.ones(len(dftest))
    dftest.insert(0, 'ones', addones, allow_duplicates=True)
    x_test = dftest.to_numpy()

    total_error = 0

    for i in range(len(dftest)):
        xi = x_test[i, :-1]
        yi = x_test[i, -1]
        prediction = np.dot(xi, w)
        error = yi - prediction
        total_error += 0.5 * error ** 2

    return total_error

w, cost_values, count = batch_gradient_descent(df, features)
test_error = calculate_error(w,dftest)
print('test error: ',test_error)

print('iterations: ', count)
print('w:', w)

def plot_cost(cost_values):
    plt.plot(cost_values)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Cost vs. Iteration')
    plt.show()

plot_cost(cost_values)

