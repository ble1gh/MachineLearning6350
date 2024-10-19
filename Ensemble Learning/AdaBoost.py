#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 4 11:20:35 2024

@author: brookleigh
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

#choose method of data splitting:
method = 'EN' #information gain (entropy)
#method = 'ME' #majority error
#method = 'GI'#gini-index

#maximum tree depth
maxdepth = 2

#number of iterations:
max_T = 10

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
df = pd.read_csv('/Users/brookleigh/Documents/GitHub/MachineLearning6350/Ensemble Learning/bank/train.csv', names=cols)
dftest = pd.read_csv('/Users/brookleigh/Documents/GitHub/MachineLearning6350/Ensemble Learning/bank/test.csv', names=cols)


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
        Sy = S['label']
        weights = S['weights']
        total_weight = np.sum(weights)
        entropy = []
        
        for val in Sy.unique():
            label_weight = np.sum(weights[Sy == val])
            proportion = label_weight / total_weight
            value_entropy = -proportion*np.log2(proportion)
            entropy.append(value_entropy)
        ent = sum(entropy)
        return(ent)
    
    def majority_error(self,S):
        Sy = S['label']
        label_counts = Sy.value_counts()
        majority_class_count = label_counts.max()
        majority_error = 1-(majority_class_count/len(Sy))
        return majority_error
    
    def gini_index(self,S):
        Sy = S['label']
        label_counts = Sy.value_counts()
        sum = 0
        for count in label_counts:
            sum += (count/len(S))**2
        gini_idx = 1-sum
        return gini_idx
    
    #caculates "information gain" depending on requested method
    def info_gain(self,df,col,Attributes,Method):
        num = 0
        Sv = df[[col, 'label', 'weights']]
        total_weight = np.sum(df['weights'])

        
        if Attributes[col] == 'numeric':
            num = 1
            median = Sv[col].median()
            df['numeric'] = np.where(df[col] > median,True, False)
            Sv = df[['numeric', 'label', 'weights']]
        if Method == 'EN':
            sum = 0;
            if num == 1:
                for j in [True, False]:
                    e = Sv.loc[Sv['numeric'] == j]
                    e_weights = np.sum(e['weights'])
                    sum += (e_weights)*self.entropy(e)
            else:
                for v in Attributes[col]:
                    e = Sv.loc[Sv[col] == v]
                    e_weights = np.sum(e['weights'])
                    sum += (e_weights)*self.entropy(e)
            infogain = self.entropy(Sv)-sum
            return(infogain)
        
        if Method == 'ME':
            sum = 0;
            if num == 1:
                for j in [True, False]:
                    e = Sv.loc[Sv['numeric'] == j]
                    if len(e) != 0:
                        sum  = sum+(len(e)/len(df))*self.majority_error(e)
            else:
                for v in Attributes[col]:
                    e = Sv.loc[Sv[col] == v]
                    if len(e) != 0:
                        sum = sum+(len(e)/len(df))*self.majority_error(e)
            majorityerror = self.majority_error(Sv)-sum
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
            median = df[col].median()
            df['numeric'] = np.where(df[col] > median,True, False)
        if num == 1:
            for j in [True, False]:
                Svi = df.index[df['numeric'] == j]
                Sv.append(Svi)
        else:
            for v in A[col]:
                Svi = df.index[df[col]==v]
                Sv.append(Svi)
        cols.remove(col)
        return(Sv,cols)
    
    def is_unique(self,s):
        a = s.to_numpy()
        return (a[0] == a).all()
    
    def build_tree(self,df,A,Attributes,Label,current_depth,Method):
            
        if current_depth >= maxdepth or self.is_unique(df['label']) or A == []:
            val = df.label.mode()[0]
            return Node(value=val)
        else:
            maxgain = -1
            splitter = None
            for Att in A:
                i = self.info_gain(df, Att, Attributes, Method)
                if i > maxgain:
                    maxgain = i
                    splitter = Att
            Svi, remaining_cols = self.split(df,splitter,A,Attributes)
            
            children = []
            for i,idx in enumerate(Svi):
                if idx.empty:
                    val = df['label'].mode()[0]  # Handle empty subsets
                    child = Node(value=val)
                else:
                    child = self.build_tree(df.loc[idx],remaining_cols, Attributes, Label, current_depth+1,method)
                    children.append(child)
            if Attributes[splitter] == 'numeric':
                median = df[splitter].median()
                return Node(splitter,len(Svi),maxgain,None,children,median)
            else:
                return Node(splitter,len(Svi),maxgain,None,children)
    
    #predict value of a given row recursively
    def predict(self,node,row):
        if node.value != None:
            return node.value
        
        #datapoint value for attribute that his node decides on
        value = row[node.attributesplit]
        attribute_vals = self.Attributes[node.attributesplit]
        
        for i, child in enumerate(node.children):
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
        for i, row in df.iterrows():
            actual_label = row['label']
            if actual_label == 'yes':
                actual_label = 1
            else:
                actual_label = -1
            
            predicted_label = self.predict(tree, row)
            if predicted_label == 'yes':
                predicted_label = 1
            else:
                predicted_label = -1

            running_error += row['weights']*actual_label*predicted_label

        # Calculate and return the training error as a fraction of misclassified samples
        training_error = 0.5-0.5*running_error
        return training_error


class adaboost():
    def __init__(self,T):
        self.T = T
    
    #build model
    def build_model(self,df,dftest,Attributes,A,Label):
        
        self.mytree = Tree(maxdepth, Attributes, Label)
        
        #intialize weights
        df['weights'] = 1 / len(df)
        dftest['weights'] = 1 / len(df)
        df_unweighted = df.copy()
        
        results = []
        trees = []

        for i in range(0,self.T):
            #print(i)
            a = self.mytree.build_tree(df, A, Attributes, Label, 1, method)
            trees.append(a)
            error = self.mytree.calc_error(a,df)
            training_error = self.mytree.calc_error(a,df_unweighted)
            test_error = self.mytree.calc_error(a,dftest)
            
            #print('error: ',error)
            alpha = 0.5*np.log((1-error)/error)
            results.append([error, alpha,a,training_error,test_error])
            
            #update weights
            w = []
            for idx, row in df.iterrows():
                if self.mytree.predict(a,row) == 'yes': predicted_label = 1 
                else: predicted_label = -1
                #print('prediction: ',predicted_label)
                
                if row['label'] == 'yes': actual_label = 1
                else: actual_label = -1
                #print('actual: ',actual_label)
                w.append(np.exp(-1*alpha*predicted_label*actual_label))
            
            #normalize weights
            z = np.sum(w)
            df['weights'] = w / z
            #print('sum of weights: ',np.sum(weights))
            
            #update weights column
            #df['weights'] = weights
    
        df_results = pd.DataFrame(data = results, columns = ['error','alpha','tree','training error','test error'])
        #print(df_results)
        return(df_results)

    #use model on a datapoint
    def predict(self,model,row):
        prediction = 0
        for w in model.iterrows():
            if self.mytree.predict(w[1]['tree'],row) == 'yes': pred = 1 
            else: pred = -1
            prediction += w[1]['alpha']*pred
        
        if prediction > 0:
            return(1)
        elif prediction < 0:
            return(-1)
        else:
            return(0)
    
    #calculate prediction error
    def calc_error(self,results,df):
        model = results[['alpha','tree']]
        
        incorrect_predictions = 0
        for _, row in df.iterrows():
            pred = self.predict(model, row)
            
            actual_label = 1 if row['label'] == 'yes' else -1
            
            if pred != actual_label:
                incorrect_predictions += 1
                #print('incorrect prediction on row: ',i)
        prediction_error = incorrect_predictions/len(df)
        return(prediction_error)
        

training_errors = []
test_errors = []
T_range = range(1,max_T)
stump_training_errors =[]
stump_test_errors = []

def run_adaboost_iteration(T, df, dftest, Attributes, A, Label):
    print('T = ', T)
    model = adaboost(T)
    results = model.build_model(df, dftest, Attributes, A, Label)

    stump_training_error = results['training error']
    stump_test_error = results['test error']
    
    training_error = model.calc_error(results, df)
    test_error = model.calc_error(results, dftest)
    
    print(f'Iteration {T}:')
    print(f'Training Error: {training_error}')
    print(f'Test Error: {test_error}')
    print(f'Weights: {df["weights"].values}')
    
    return (stump_training_error, stump_test_error, training_error, test_error)

results = Parallel(n_jobs=-1)(delayed(run_adaboost_iteration)(T, df, dftest, Attributes, A, Label) for T in T_range)

for stump_training_error, stump_test_error, training_error, test_error in results:
    stump_training_errors.append(stump_training_error)
    stump_test_errors.append(stump_test_error)
    training_errors.append(training_error)
    test_errors.append(test_error)
    
print('training_errors:',stump_training_errors)
print('test_errors:',stump_test_errors)
# Plot training and test errors on one plot
plt.figure(figsize=(10, 5))
plt.plot(T_range, training_errors, marker='o', linestyle='-', color='blue', label='Training Error')
plt.plot(T_range, test_errors, marker='o', linestyle='-', color='green', label='Test Error')
plt.title('Training and Test Error vs Number of Iterations (T)')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
#plt.show()

# Plot stump training and test errors on one plot
plt.figure(figsize=(10, 5))

# Flatten the list of lists for stump errors
flat_stump_training_errors = [item for sublist in stump_training_errors for item in sublist]
flat_stump_test_errors = [item for sublist in stump_test_errors for item in sublist]

# Create a range for the x-axis based on the length of the flattened lists
stump_T_range = range(1, len(flat_stump_training_errors) + 1)

plt.plot(stump_T_range, flat_stump_training_errors, marker='o', linestyle='-', color='red', label='Stump Training Error')
plt.plot(stump_T_range, flat_stump_test_errors, marker='o', linestyle='-', color='orange', label='Stump Test Error')
plt.title('Stump Training and Test Error vs Number of Iterations (T)')
plt.xlabel('Number of Iterations (T)')
plt.ylabel('Error')
plt.grid(True)
plt.legend()
plt.show()



