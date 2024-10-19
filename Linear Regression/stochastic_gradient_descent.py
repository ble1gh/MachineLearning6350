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
r = 0.001

#error threshold
epsilon = 1e-6

features = ['Cement', 'Slag', 'Fly Ash', 'Water','SP', 'Coarse Aggregate', 'Fine Aggregate', 'label']
features_test = ['Cement', 'Slag', 'Fly Ash', 'Water','SP', 'Coarse Aggregate', 'Fine Aggregate']

#read in data
df = pd.read_csv('concrete/train.csv', names=features)
dftest = pd.read_csv('concrete/test.csv', names=features)
addones = np.ones(len(dftest))
dftest.insert(0, 'ones', addones, allow_duplicates=True)

def stoch_gradient_descent(df,features, learning_rate=r,epsilon=epsilon):
    addones = np.ones(len(df))
    df.insert(0, 'ones', addones, allow_duplicates=True)
    x = df.to_numpy()

    # Initialize w vector
    w = np.zeros(len(features))

    cost_diff = 1
    count = 0
    cost_values = [calculate_error(w, df)]

    while abs(cost_diff) > epsilon:
        i = np.random.randint(0, len(df))
        xi = x[i, :-1]
        yi = x[i, -1]
        e = yi - np.dot(xi, w)
        grad = -e * xi

        w_previous = w.copy()
        w -= grad * learning_rate
        
        cost = calculate_error(w,df)
        cost_diff = cost - cost_values[-1]
        cost_values.append(cost)
        count += 1

    return w, cost_values, count



def calculate_error(w, dftest):
    
    x_test = dftest.to_numpy()

    total_error = 0

    for i in range(len(dftest)):
        xi = x_test[i, :-1]
        #print('xi:',xi)
        yi = x_test[i, -1]
        #print('yi',yi)
        prediction = np.dot(xi, w)
        error = yi - prediction
        total_error += 0.5 * error ** 2

    return total_error

w, cost_values, count = stoch_gradient_descent(df, features)
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

addones = np.ones(len(df))
df.insert(0, 'ones', addones, allow_duplicates=True)
x = df.to_numpy()
y = x[:,-1]
x = x[:, :-1]
w_star = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y
print('w star: ',w_star)