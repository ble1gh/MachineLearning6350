#Machine Learning HW 4 Q3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.close('all')

#hyperparameters
C = 700/873

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
def SVM_dual(df, C):
    
    #add ones to x
    x = df.copy()
    # addones = np.ones(len(x))
    # x.insert(0, 'ones', addones, allow_duplicates=True)
    x = x.to_numpy()

    #get labels
    y = df['label'].to_numpy()

    #remove label from x
    x = x[:, :-1]
    
    #set N
    N = len(x)
    
    #initialize alpha
    alpha = np.zeros(N)
    
    #initialize K and Q
    K = np.dot(x, x.T)
    Q = np.outer(y, y)*K
    
    #objective function
    def objective(alpha):
        return 1/2*np.dot(alpha, np.dot(alpha, Q)) - np.sum(alpha)
    
    #constraint
    def constraint(alpha):
        return np.dot(alpha, y)
    
    #set bounds
    bounds = [(0, C) for i in range(N)]
    
    #set constraint
    con = {'type': 'eq', 'fun': constraint}
    
    #minimize
    res = minimize(objective, alpha, bounds=bounds, constraints=con, method='SLSQP')
    
    #get alpha
    alpha = res.x

    #find support vectors
    sv = alpha > 1e-5

    #get w and b
    w = 0
    for i in range(N):
        w += alpha[i]*y[i]*x[i]
    b = np.mean(y[sv] - np.dot(x[sv], w))
    # w = np.sum(alpha[sv]*y[sv].to_numpy().reshape(-1, 1)*x[sv], axis=0)
    # b = np.mean(y[sv].to_numpy().reshape(-1, 1) - np.dot(x[sv], w))
    
    return w,b,sv

#calculate error
def calculate_error(w, b, dftest):
    x_test = dftest.copy()
    # addones = np.ones(len(x_test))
    # x_test.insert(0, 'ones', addones, allow_duplicates=True)
    x_test = x_test.to_numpy()
    
    y_test = x_test[:, -1]
    x_test = x_test[:, :-1]
    
    y_pred = np.dot(x_test, w)+b
    y_pred = np.where(y_pred > 0, 1, -1)
    
    error = np.mean(y_pred != y_test)
    
    return error

#train SVM
w,b,sv = SVM_dual(df, C)
print('Number of Support Vectors:', np.sum(sv))
print('Support Vectors:')
print(df[sv])

#calculate error
test_error = calculate_error(w, b, dftest)
print('Test Error: ', test_error)
train_error = calculate_error(w, b, df)
print('Train Error: ', train_error)
print('w:', w)
print('b:', b)


