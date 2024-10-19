#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 19:26:33 2024

@author: brookleigh
"""

import numpy as np
import pandas as pd
from random import choices
import time
from collections import Counter
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#choose method of data splitting:
method = 'EN' #information gain (entropy)
#method = 'ME' #majority error
#method = 'GI'#gini-index

#maximum tree depth
maxdepth = 16

#set number of iterations
max_T = 100

G_size = 6

#set percentage of full data set used to train each tree
m_size = 0.1

S = []
Label = ['unacc', 'acc', 'good', 'vgood']
A =[]
cols = []


#input data description
Attributes = {'age':'numeric',
              'job': ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student","blue-collar","self-employed","retired","technician","services" ],
              'marital':["married","divorced","single"],
              'education': ["unknown","secondary","primary","tertiary"],
              'default': ["yes","no"],
              'balance':'numeric',
              'housing': ["yes","no"],
              'loan': ["yes","no"],
              'contact': ["unknown","telephone","cellular"],
              'day': 'numeric',
              'month': ["jan", "feb", "mar","apr","may","jun","jul","aug","sep","oct", "nov", "dec"],
              'duration':'numeric',
              'campaign': 'numeric',
              'pdays': 'numeric',
              'previous':'numeric',
              'poutcome': ["unknown","other","failure","success"],
              'label': ["yes", "no"]
    }

for Att in Attributes:
    cols.append(Att)
A = cols[0:-1]

    
df = pd.read_csv('/Users/brookleigh/Documents/GitHub/MachineLearning6350/Ensemble Learning/bank/train.csv', names=cols, header=0)

#replace unknown values with the most common known value in that category
for col in df:
    #print(col)
    #print(Attributes[col])
    if "unknown" in Attributes[col]:
        #print(df[col])
        known_col = df[col].loc[df[col] != "unknown"]
        #print(known_col)
        counts = known_col.value_counts()
        majority_idx = counts.idxmax()
        #print('majority_idx: ',majority_idx)
        df.replace({col:"unknown"},majority_idx,inplace=True)
        #print(df[col])
        
#print(df)
dftest = pd.read_csv('/Users/brookleigh/Documents/GitHub/MachineLearning6350/Ensemble Learning/bank/test.csv', names=cols, header=0)

for col in dftest:
    #print(col)
    #print(Attributes[col])
    if "unknown" in Attributes[col]:
        #print(df[col])
        known_col = dftest[col].loc[dftest[col] != "unknown"]
        #print(known_col)
        counts = known_col.value_counts()
        majority_idx = counts.idxmax()
        #print('majority_idx: ',majority_idx)
        dftest.replace({col:"unknown"},majority_idx,inplace=True)


for Att in Attributes:
    if "unknown" in Attributes[Att]:
        Attributes[Att].remove("unknown")
#print(df)
#X = df.drop(columns=['label'])

#make a Node class
class Node():
    def __init__(self,attributesplit=None,num_branches=None,infogain=None,value=None,children=None,divider=None):
        
        #for decision node:
        self.attributesplit = attributesplit
        self.num_branches = num_branches
        self.infogain = infogain
        self.children = children
        
        #for numerical category
        self.divider = divider
        
        #for leaf node:
        self.value = value
        
#make a tree class
class Tree():
    def __init__(self, depth,Attributes,Label):
        
        #initialize root node
        self.root = None
        
        #stopping condition
        self.maxdepth = depth
        
        self.Attributes = Attributes
        self.Label = Label
    
        
    #function calculates entropy given a data set and a set of labels
    def entropy(self,S):
        #print('S in entropy')
        #print(S)
        Sy = S['label']
        #print(Sy)
        entropy = []
        for val in Sy.unique():
            #print(val)
            e = Sy[Sy == val]
            #print(e)
            value_entropy = -1*(len(e)/len(Sy))*np.log2(len(e)/len(Sy))
            entropy.append(value_entropy)
            #print('value_entropy')
            #print(value_entropy)
        ent = sum(entropy)
        return(ent)
    
    def majority_error(self,S):
        Sy = S['label']
        #print(Sy)
        label_counts = Sy.value_counts()
        majority_class_count = label_counts.max()
        #print(label_counts)
        #print(majority_class_count)
        majority_error = 1-(majority_class_count/len(Sy))
        #print('majority error ')
        #print(majority_error)
        return majority_error
    
    def gini_index(self,S):
        Sy = S['label']
        #print(Sy)
        label_counts = Sy.value_counts()
        #print(label_counts)
        sum = 0
        for count in label_counts:
            sum += (count/len(S))**2
        gini_idx = 1-sum
        return gini_idx
    
    #caculates "information gain" depending on requested method
    def info_gain(self,df,col,Attributes,Method):
        #print(df)
        #print('col: ',col)
        num = 0
        Sv = df[[col, 'label']]
        if Attributes[col] == 'numeric':
            num = 1
            median = Sv[col].median()
            #print(median)
            df['numeric'] = np.where(df[col] > median,True, False)
            #Svnum = pd.DataFrame(data=compare_median,columns=col)
            #Svnum = [compare_median, Sv['label']]
            #print(df)
            Sv = df[['numeric', 'label']]
        #print('Sv: ',Sv)
        if Method == 'EN':
            sum = 0;
            if num == 1:
                for j in [True, False]:
                    e = Sv.loc[Sv['numeric'] == j]
                    sum = sum+(len(e)/len(df))*self.entropy(e)
            else:
                for v in Attributes[col]:
                    e = Sv.loc[Sv[col] == v]
                    #print('v:',v)
                    #print('e: ',e)
                    sum = sum+(len(e)/len(df))*self.entropy(e)
                    #print('current sum: ')
                    #print(sum)
            infogain = self.entropy(Sv)-sum
            #print(infogain)
            return(infogain)
        
        if Method == 'ME':
            sum = 0;
            #print(Attributes[col])
            if num == 1:
                for j in [True, False]:
                    e = Sv.loc[Sv['numeric'] == j]
                    if len(e) != 0:
                        sum  = sum+(len(e)/len(df))*self.majority_error(e)
            else:
                for v in Attributes[col]:
                    e = Sv.loc[Sv[col] == v]
                    #print('v:',v)
                    #print('e: ',e)
                    if len(e) != 0:
                        sum = sum+(len(e)/len(df))*self.majority_error(e)
                    #print('current sum: ')
                    #print(sum)
            #print(self.majority_error(Sv))
            majorityerror = self.majority_error(Sv)-sum
            #print(majorityerror)
            return(majorityerror)
        
        if Method == 'GI':
            gini_index = self.gini_index(Sv)
            return(gini_index)
            
    #find indeces of all rows in main dataframe given an attribute and value, return
    #a list of index sets for each value the attribute can take, and a new list of 
    #attributes with that one removed
    def split(self,df,col,cols,A):
        Sv = []
        num = 0
        if Attributes[col] == 'numeric':
            num = 1
            #print(col)
            median = df[col].median()
            #print(median)
            df['numeric'] = np.where(df[col] > median,True, False)
            #Svnum = pd.DataFrame(data=compare_median,columns=col)
            #Svnum = [compare_median, Sv['label']]
            #print(df)
            #Sv = df[['numeric', 'label']]
        #print('col: ',col)
        #print(A[col])
        if num == 1:
            for j in [True, False]:
                Svi = df.index[df['numeric'] == j]
                Sv.append(Svi)
        else:
            for v in A[col]:
                Svi = df.index[df[col]==v]
                Sv.append(Svi)
                #Sv = df.loc[Svi]
        cols.remove(col)
        #print(cols)
        return(Sv,cols)
    
    def is_unique(self,s):
        a = s.to_numpy()
        return (a[0] == a).all()
    
    #print(is_unique(df['label']))
    
    
    def build_tree(self,df,A,Attributes,Label,current_depth,Method,G_size):
            
        #print('current depth: ',current_depth)
        if current_depth >= maxdepth or self.is_unique(df['label']) or A == []:
            val = df.label.mode()[0]
            return Node(value=val)
        else:
            maxgain = -1
            splitter = None
            #print(A)
            G = choices(A,k=G_size)
            for Att in G:
                #print(Att)
                #print('A: ',A)
                #print('Att: ',Att)
                i = self.info_gain(df, Att, Attributes, Method)
                #print('i: ',i)
                #print('maxgain: ',maxgain)
                if i > maxgain:
                    maxgain = i
                    splitter = Att
            #print(splitter)
            Svi, remaining_cols = self.split(df,splitter,A,Attributes)
            #print('Svi: ',Svi)
            
            children = []
            for i,idx in enumerate(Svi):
                #print(df.loc[i])
                #print(Attributes[splitter])
                if idx.empty:
                    val = df['label'].mode()[0]  # Handle empty subsets
                    child = Node(value=val)
                else:
                    child = self.build_tree(df.loc[idx],remaining_cols, Attributes, Label, current_depth+1,method,G_size)
                    children.append(child)
            if Attributes[splitter] == 'numeric':
                median = df[splitter].median()
                return Node(splitter,len(Svi),maxgain,None,children,median)
            else:
                return Node(splitter,len(Svi),maxgain,None,children)
    
    #predict value of a given row recursively
    def predict(self,node,row):
        if node.value != None:
            #print(node.value)
            return node.value
        
        #datapoint value for attribute that this node decides on
        #print(node.attributesplit)
        splitter = node.attributesplit
        #print('splitter: ',splitter)
        #print('row type: ',type(row))
        #print('row: ',row)
        value = pd.Series(row[1])[node.attributesplit]
        attribute_vals = Attributes[splitter]
        
        
        for i, child in enumerate(node.children):
            #print('attribute_vals: ',attribute_vals)
            #print('value: ',value)
            if attribute_vals == 'numeric':
                if i == 0 and value>node.divider:
                    return self.predict(child, row)
                else:
                    return self.predict(child, row)
            else:
                if value == attribute_vals[i]:
                    return self.predict(child,row)
    
    #calculate prediction errror over a given dataset
    def calc_error(self,tree,df):
        incorrect_predictions = 0
        total_samples = len(df)
        
        # iterate over each row in the dataframe
        for idx, row in df.iterrows():
            actual_label = row['label']
            #print('actual: ',actual_label)
            predicted_label = self.predict(tree, row)
            #print('prediction: ',predicted_label)

            if predicted_label != actual_label:
                incorrect_predictions += 1

        # Calculate and return the training error as a fraction of misclassified samples
        training_error = incorrect_predictions / total_samples
        #print(incorrect_predictions)
        return training_error

class Bagging():
    def __init__(self,T):
        self.T = T

    def baggem(self,df,m_size):
        #choosing my bootstrap sample size
        m = int(len(df)*m_size)
        
        #range of df indexes for bootstrap sample
        df_range = range(0,len(df)-1)
        
        trees = []
        training_error = []
        test_error = []
        
        for i in range(self.T):
            #start_time = time.time()
            idx = choices(df_range, k=m)
            S = pd.DataFrame(df.iloc[idx]).reset_index(drop=True)
            
            # Build tree on the bootstrap sample
            self.mytree = Tree(Attributes, Label, maxdepth)
            a = self.mytree.build_tree(S, A, Attributes, Label, 1, method,G_size)
            trees.append(a)
            
        # Calculate and print errors
        training_error, training_bias, train_var = self.calc_error(trees, df)
        test_error, test_bias, test_var = self.calc_error(trees, dftest)

        
        
        return training_error, test_error, training_bias, train_var, test_var
    
    def predict(self,trees,row):
        predictions = []
        for tree in trees:
            pred = self.mytree.predict(tree, row)
            pred = 1 if pred == 'yes' else 0
            predictions.append(pred)
        
        average = np.average(predictions)

        v_squared = 0
        for pred in predictions:
            v_squared += (pred-average)**2

        if len(predictions) > 1:
            var = np.sqrt(1/(len(predictions)-1)*v_squared)
        else:
            var = 0

        final_prediction = round(average,1)
        #most_common_value = Counter(predictions).most_common(1)[0][0]
        return final_prediction,var
    
    def calc_error(self,trees,df):
        incorrect_predictions = 0
        #print(df)
        for row in df.iterrows():
            #print('row in calc_error:',row)
            pred,var = self.predict(trees, row)
            actual_label = 1 if pd.Series(row[1])['label'] == 'yes' else 0
            if pred != actual_label:
                incorrect_predictions += 1
            
        
        error = incorrect_predictions/len(df)
        bias = (incorrect_predictions)**2/len(df)
        
        return error,bias,var

def run_bagging(T, df, dftest, Attributes, Label, maxdepth, method, G_size, m_size):
    bagged = Bagging(T)
    return bagged.baggem(df, m_size)

results = Parallel(n_jobs=-1)(delayed(run_bagging)(T, df, dftest, Attributes, Label, maxdepth, method, G_size, m_size) for T in range(1, max_T))

training_errors, test_errors, train_biases, train_vars, test_vars = zip(*results)
#print('training_error: ',training_errors)
#print('test_error: ',test_errors)

print('training bias last: ',train_biases[-1])
print('training variance last: ',train_vars[-1])

#plot results
T_range = range(1,max_T)

plt.figure(figsize=(10, 5))
plt.plot(T_range, test_errors, marker='o', linestyle='-', color='green', label='Test Error')
plt.plot(T_range, training_errors, marker='o', linestyle='-', color='blue', label='Training Error')
plt.title('Random Forest Test Error vs Number of Iterations (T) - Feature subset: 6')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Test Error')
plt.grid(True)
plt.legend()
plt.show()

    






