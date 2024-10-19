#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 4 11:20:35 2024

@author: brookleigh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import choices
import time

#choose method of data splitting:
method = 'EN' #information gain (entropy)
#method = 'ME' #majority error
#method = 'GI'#gini-index

#maximum tree depth
maxdepth = 10

#number of iterations:
max_T = 2

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

#read in data
df = pd.read_csv('bank/train.csv',names=cols)

#intialize weights
df['weights'] = 1 / len(df)

#test data
dftest = pd.read_csv('bank/test.csv',names=cols)
dftest['weights'] = 1 / len(dftest)


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
    def __init__(self,Attributes,Label,maxdepth):
        
        #initialize root node
        self.root = None
        
        self.max_depth = maxdepth
        
        self.Attributes = Attributes
        self.Label = Label
    
        
    #function calculates entropy given a data set and a set of labels
    def entropy(self,S):
        #print('S in entropy')
        #print(S)
        Sy = S['label']
        weights = S['weights']
        total_weight = np.sum(weights)
        entropy = []
        
        for val in Sy.unique():
            label_weight = np.sum(weights[Sy == val])
            proportion = label_weight / total_weight
            #print(val)
            value_entropy = -proportion*np.log2(proportion)
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
        Sv = df[[col, 'label', 'weights']]
        total_weight = np.sum(df['weights'])
        
        if Attributes[col] == 'numeric':
            num = 1
            median = Sv[col].median()
            #print(median)
            df['numeric'] = np.where(df[col] > median,True, False)
            #Svnum = pd.DataFrame(data=compare_median,columns=col)
            #Svnum = [compare_median, Sv['label']]
            #print(df)
            Sv = df[['numeric', 'label', 'weights']]
        #print('Sv: ',Sv)
        if Method == 'EN':
            sum = 0;
            if num == 1:
                for j in [True, False]:
                    e = Sv.loc[Sv['numeric'] == j]
                    e_weights = np.sum(e['weights'])
                    sum += (e_weights/total_weight)*self.entropy(e)
            else:
                for v in Attributes[col]:
                    e = Sv.loc[Sv[col] == v]
                    e_weights = np.sum(e['weights'])
                    #print('v:',v)
                    #print('e: ',e)
                    sum += (e_weights/total_weight)*self.entropy(e)
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
    def split(self, df, col, cols):
        Sv = []
        if Attributes[col] == 'numeric':
            median = df[col].median()
            df['numeric'] = np.where(df[col] > median, True, False)
            for j in [True, False]:
                Svi = df.index[df['numeric'] == j]
                Sv.append(Svi)
        else:
            for v in Attributes[col]:
                Svi = df.index[df[col] == v]
                Sv.append(Svi)
        
        cols.remove(col)  # Remove the used attribute
        #print('After splitting on:', col, 'Remaining cols:', cols)
        return Sv, cols
    
    def is_unique(self, s):
        return s.nunique() == 1  # Checks if all labels are the same
    
    #print(is_unique(df['label']))
    
    
    def build_tree(self, df, A, Attributes, Label, current_depth, Method):
        # Add check for depth and stopping criteria
        if current_depth >= self.max_depth or self.is_unique(df['label']) or len(A) == 0:
            val = df['label'].mode()[0]
            return Node(value=val)
        
        # Find the best attribute to split on
        maxgain = -1
        splitter = None
        for Att in A:
            i = self.info_gain(df, Att, Attributes, Method)
            if i > maxgain:
                maxgain = i
                splitter = Att
        
        if splitter is None:
            # If no valid splitter is found, return majority label
            val = df['label'].mode()[0]
            return Node(value=val)
        
        # Perform the split
        Svi, A_copy = self.split(df, splitter, A.copy())  # Pass a copy of A
        if len(Svi) == 0:  # If split results in no data, stop recursion
            val = df['label'].mode()[0]
            return Node(value=val)
    
        # Recursive call to build children
        children = []
        for idx in Svi:
            if idx.empty:
                val = df['label'].mode()[0]
                child = Node(value=val)
            else:
                child = self.build_tree(df.loc[idx], A_copy, Attributes, Label, current_depth + 1, Method)
            children.append(child)
        
        if Attributes[splitter] == 'numeric':
            median = df[splitter].median()
            return Node(splitter, len(Svi), maxgain, None, children, median)
        else:
            return Node(splitter, len(Svi), maxgain, None, children)

    
    #predict value of a given row recursively
    def predict(self,node,row):
        if node.value != None:
            #print(node.value)
            return node.value
        
        #datapoint value for attribute that his node decides on
        value = row[node.attributesplit]
        attribute_vals = self.Attributes[node.attributesplit]
        
        
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
        running_error = 0
        
        # iterate over each row in the dataframe
        for idx, row in df.iterrows():
            actual_label = row['label']
            if actual_label == 'yes':
                actual_label = 1
            else:
                actual_label = -1
            #print('actual: ',actual_label)
            
            predicted_label = self.predict(tree, row)
            if predicted_label == 'yes':
                predicted_label = 1
            else:
                predicted_label = -1
            #print('prediction: ',predicted_label)

            running_error += row['weights']*actual_label*predicted_label

        # Calculate and return the training error as a fraction of misclassified samples
        training_error = 0.5-0.5*running_error
        #print(incorrect_predictions)
        return training_error
        
#choosing my bootstrap sample size
m = int(len(df)/2)

#range of df indexes for bootstrap sample
df_range = range(0,len(df)-1)

for i in range(max_T):
    start_time = time.time()
    print(f'T= {i}')
    idx = choices(df_range, k=m)
    S = pd.DataFrame(df.iloc[idx]).reset_index(drop=True)
    print(f'Bootstrap sample size: {len(S)}')
    
    # Build tree on the bootstrap sample
    mytree = Tree(Attributes, Label, maxdepth)
    a = mytree.build_tree(S, A, Attributes, Label, 1, method)
    
    # Calculate and print errors
    error = mytree.calc_error(a, df)
    print(f'Error after iteration {i}: {error}')
    print(f'Iteration {i} took {time.time() - start_time} seconds')
    


# # Plot training errors
# plt.figure(figsize=(10, 5))
# plt.plot(T_range, training_errors, marker='o', linestyle='-', color='blue', label='Training Error')
# plt.title('Training Error vs Number of Iterations (T)')
# plt.xlabel('Number of Iterations (T)')
# plt.ylabel('Training Error')
# plt.grid(True)
# plt.legend()
# plt.show()

# # Plot test errors
# plt.figure(figsize=(10, 5))
# plt.plot(T_range, test_errors, marker='o', linestyle='-', color='green', label='Test Error')
# plt.title('Test Error vs Number of Iterations (T)')
# plt.xlabel('Number of Iterations (T)')
# plt.ylabel('Test Error')
# plt.grid(True)
# plt.legend()
# plt.show()




