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
    m = 0

    train_error = []
    c = 1
    for t in range(T):
        weights = []
        

        #shuffle data
        np.random.shuffle(x)

        #train
        for i in range(len(x)):
            xi = x[i, :-1]
            yi = x[i, -1]
            if yi * np.dot(xi, w) <= 0:
                #print('prediction:', np.dot(xi, w), ', yi:', yi)
                weights.append([w.copy(),c])
                w += r * yi * xi
                m += 1
                c = 1
            else:
                c += 1
        
        #for error tracking
        if t > 0:
            error = calculate_error(weights, df)
            train_error.append(error)

    return weights,train_error

def calculate_error(weights, dftest):
    x_test = dftest.copy()
    addones = np.ones(len(x_test))
    x_test.insert(0, 'ones', addones, allow_duplicates=True)
    x_test = x_test.to_numpy()

    total_error = 0

    for i in range(len(dftest)):
        xi = x_test[i, :-1]
        yi = x_test[i, -1]
        sum = 0

        for j in range(len(weights)):
            w = weights[j][0]
            c = weights[j][1]
            sum += c* np.dot(xi, w)
        if sum*yi <= 0:
            total_error += 1

    error = total_error / len(dftest)
    return error

w,train_error = standard_perceptron(df, features, T)
print('w:', w)
prediction_error = calculate_error(w, dftest)
print('Prediction error:', prediction_error)

T_plot = range(T)
T_plot = T_plot[1:]

plt.plot(T_plot, train_error, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Training Error')
plt.title('Training Error vs Epochs')
plt.show()