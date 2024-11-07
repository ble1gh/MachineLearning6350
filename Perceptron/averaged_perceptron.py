#Machine Learning HW 3 part 2a

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

#data location
train_data = '/Users/brookleigh/Documents/GitHub/MachineLearning6350/Perceptron/bank-note/train.csv'
test_data = '/Users/brookleigh/Documents/GitHub/MachineLearning6350/Perceptron/bank-note/test.csv'

#number of epochs
T = 10

#learning rate
r = 1

features = ['waveletVar', 'waveletSkew', 'waveletCurt', 'imageEntropy', 'label']

#read in data
df = pd.read_csv(train_data, names=features)
dftest = pd.read_csv(test_data, names=features)

#convert labels to -1 and 1
df['label'] = df['label'].apply(lambda x: -1 if x == 0 else 1)
dftest['label'] = dftest['label'].apply(lambda x: -1 if x == 0 else 1)

#train perceptron
def standard_perceptron(df, features, T):
    x = df.copy()
    addones = np.ones(len(x))
    x.insert(0, 'ones', addones, allow_duplicates=True)
    x = x.to_numpy()

    # Initialize w vector
    w = np.zeros(len(features))
    a = w.copy()

    train_error = []
    for t in range(T):

        if t > 0:
            error = calculate_error(w, df)
            train_error.append(error)

        #shuffle data
        np.random.shuffle(x)

        for i in range(len(x)):
            xi = x[i, :-1]
            yi = x[i, -1]
            if yi * np.dot(xi, w) <= 0:
                #print('prediction:', np.dot(xi, w), ', yi:', yi)
                w += r * yi * xi
            a += w

    return a,train_error

def calculate_error(w, dftest):
    x_test = dftest.copy()
    addones = np.ones(len(x_test))
    x_test.insert(0, 'ones', addones, allow_duplicates=True)
    x_test = x_test.to_numpy()

    total_error = 0

    for i in range(len(dftest)):
        xi = x_test[i, :-1]
        yi = x_test[i, -1]
        if yi * np.dot(xi, w) <= 0:
            total_error += 1

    error = total_error / len(dftest)
    return error

a,train_error = standard_perceptron(df, features, T)
print('a:', a)
prediction_error = calculate_error(a, dftest)
print('Prediction error:', prediction_error)

T_plot = range(T)
T_plot = T_plot[1:]

plt.plot(T_plot, train_error, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Training Error')
plt.title('Training Error vs Epochs')
plt.show()