#Machine Learning HW 3 part 2a and b

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.close('all')

#hyperparameters
T = 100
C = 700/873
learning_rate_schedule = 'b'
gamma_0 = .8
a = .5

#stopping condition
epsilon = 1e-3

#data location
train_data = '/Users/brookleigh/Documents/GitHub/MachineLearning6350/SVM/bank-note/train.csv'
test_data = '/Users/brookleigh/Documents/GitHub/MachineLearning6350/SVM/bank-note/test.csv'

features = ['waveletVar', 'waveletSkew', 'waveletCurt', 'imageEntropy', 'label']

#read in data
df = pd.read_csv(train_data, names=features)
dftest = pd.read_csv(test_data, names=features)

#convert labels to -1 and 1
df['label'] = df['label'].apply(lambda x: -1 if x == 0 else 1)
dftest['label'] = dftest['label'].apply(lambda x: -1 if x == 0 else 1)

#train SVM
def SVM_primal_stochastic_subgradient(df, features, T, C, learning_rate_schedule, gamma_0, a, epsilon):

    #add ones to x
    x = df.copy()
    addones = np.ones(len(x))
    x.insert(0, 'ones', addones, allow_duplicates=True)
    x = x.to_numpy()

    #Initialize w vector
    w = np.zeros(len(features))

    #set N
    N = len(x)

    train_error = []
    for t in range(T):

        #set gamma
        if learning_rate_schedule == 'a':
            gamma = gamma_0/((gamma_0/a)*t+1)
        elif learning_rate_schedule == 'b':
            gamma = gamma_0/(t+1)
        else:
            gamma = gamma_0/(t+1)

        #shuffle data and select random row
        i = np.random.randint(0, len(x))
        xi = x[i, :-1]
        yi = x[i, -1]

        if yi*np.dot(xi, w) <= 1:
            w = (1 - gamma) * w + gamma * C * N * yi * xi
            subgrad = w-C*N*xi*yi
        else:
            w = (1 - gamma) * w
            subgrad = w

        if t > 0:
            error = calculate_error(w, df)
            train_error.append(error)

        if np.linalg.norm(subgrad) < epsilon:
            break

    return w, train_error

#calculate error
def calculate_error(w, dftest):
    x_test = dftest.copy()
    addones = np.ones(len(x_test))
    x_test.insert(0, 'ones', addones, allow_duplicates=True)
    x_test = x_test.to_numpy()

    total_error = 0
    for i in range(len(x_test)):
        xi = x_test[i, :-1]
        yi = x_test[i, -1]
        if yi * np.dot(xi, w) <= 0:
            total_error += 1

    return total_error/len(x_test)

w, train_error = SVM_primal_stochastic_subgradient(df, features, T, C, learning_rate_schedule, gamma_0, a, epsilon)
print('w:', w)
prediction_error = calculate_error(w, dftest)
print('Prediction error:', prediction_error)
final_train_error = calculate_error(w, df)
print('Final training error:', final_train_error)

#plot training error
plt.figure()
plt.plot(train_error)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.title('Training Error')
plt.show()