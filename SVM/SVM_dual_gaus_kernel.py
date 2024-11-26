#Machine Learning HW 4 Q3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.close('all')

#hyperparameters
C = 700/873
gamma = 100

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
    
    x = df.copy()
    x = x.to_numpy()

    #get labels
    y = df['label'].to_numpy()

    #remove label from x
    x = x[:, :-1]
    
    #set N
    N = len(x)
    
    #initialize alpha
    alpha = np.zeros(N)

    #calculate gaussian kernel
    def gaussian_kernel(x1, x2):
        return np.exp(-np.linalg.norm(x1-x2)**2/gamma)
    
    #initialize K
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = gaussian_kernel(x[i], x[j])
    
    #initialize Q
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
    n_sv = np.sum(sv)
    alpha_sv = alpha[sv]
    x_sv = x[sv]
    y_sv = y[sv]

    #get w and b
    w = 0
    for i in range(N):
        w += alpha[i]*y[i]*x[i]
    
    inner_sum = np.zeros(n_sv)
    for k in range(n_sv):
        for i in range(n_sv):
            inner_sum[k] += alpha_sv[i]*y_sv[i]*gaussian_kernel(x_sv[k], x_sv[i])
    b = np.mean(y[sv] - inner_sum)
    
    return w,alpha_sv,b,x_sv,y_sv,sv

#predict
def predict(x, alpha, b, x_sv, y_sv):
    y_pred = np.zeros(len(x))
    for i in range(len(x)):
        y_pred[i] = np.sum(alpha*y_sv*np.exp(-np.linalg.norm(x[i]-x_sv, axis=1)**2/gamma))+b
    y_pred = np.where(y_pred > 0, 1, -1)
    return y_pred

#calculate error
def calculate_error(df, alpha, b, dftest):
    x_test = dftest.copy()
    x_sv = df.copy()
    # addones = np.ones(len(x_test))
    # x_test.insert(0, 'ones', addones, allow_duplicates=True)
    x_test = x_test.to_numpy()
    x_sv = x_sv.to_numpy()
    
    y_test = x_test[:, -1]
    x_test = x_test[:, :-1]
    y_sv = x_sv[:, -1]
    x_sv = x_sv[:, :-1]

    y_pred = predict(x_test, alpha, b, x_sv, y_sv)
    
    error = np.mean(y_pred != y_test)
    
    return error

#train SVM
w,alpha_sv,b,x_sv,y_sv,sv = SVM_dual(df, C)
print('Number of Support Vectors:', len(alpha_sv))
print('Support Vectors:')
print(df[sv])

#calculate error
test_error = calculate_error(df[sv], alpha_sv, b, dftest)
print('Test Error: ', test_error)
train_error = calculate_error(df[sv], alpha_sv, b, df)
print('Train Error: ', train_error)
print('w:', w)
print('b:', b)

#export support vectors
#df[sv].to_csv('/Users/brookleigh/Desktop/gamma100.csv', index=True)

