#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:17:35 2024

@author: brookleigh
"""

import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

#choose method of data splitting:
method = 'EN' #information gain (entropy)
#method = 'ME' #majority error
#method = 'GI'#gini-index

#maximum tree depth
maxdepth = 6

S = []
Label = ['unacc', 'acc', 'good', 'vgood']
Attributes = {}
A =[]


#load data description
with open('car/data-desc.txt', 'r') as g:
    for line in g:
        if ':' in line:
            attribute, vals = line.split(':')
            attribute = attribute.strip()
            vals = vals.strip().strip('.').split(',')
            for i, val in enumerate(vals):
                val = val.strip()
                vals[i] = val
            Attributes[attribute] = vals
            A.append(attribute)

cols = [At for At in Attributes]
cols.append('label')

df = pd.read_csv('car/train.csv',names=cols)
#print(df)
#X = df.drop(columns=['label'])

#make a Node class
class Node():
    def __init__(self,attributesplit=None,num_branches=None,infogain=None,value=None,children=None):
        
        #for decision node:
        self.attributesplit = attributesplit
        self.num_branches = num_branches
        self.infogain = infogain
        self.children = children

        
        #for leaf node:
        self.value = value
        
#make a tree class
class Tree():
    def __init__(self, maxdepth,Attributes,Label):
        
        #initialize root node
        self.root = None
        
        #stopping condition
        self.maxdepth = maxdepth
        
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
        print(Sy)
        label_counts = Sy.value_counts()
        print(label_counts)
        sum = 0
        for count in label_counts:
            sum += (count/len(S))**2
        gini_idx = 1-sum
        return gini_idx
    
    #caculates "information gain" depending on requested method
    def info_gain(self,df,col,Attributes,Method):
        #print(df)
        #print('col: ',col)
        Sv = df[[col, 'label']]
        #print('Sv: ',Sv)
        if Method == 'EN':
            sum = 0;
            for v in Attributes[col]:
                e = Sv.loc[Sv[col] == v]
                #print('v:',v)
                #print('e: ',e)
                sum = sum+(len(e)/len(df))*self.entropy(e)
                #print('current sum: ')
                #print(sum)
            infogain = self.entropy(Sv)-sum
            return(infogain)
        
        if Method == 'ME':
            sum = 0;
            #print(Attributes[col])
            for v in Attributes[col]:
                e = Sv.loc[Sv[col] == v]
                #print('v:',v)
                #print('e: ',e)
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
        #print('col: ',col)
        #print(A[col])
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
    
    
    def build_tree(self,df,A,Attributes,Label,current_depth,Method):
            
        if current_depth >= maxdepth or self.is_unique(df['label']) or A == []:
            val = df.label.mode()[0]
            return Node(value=val)
        else:
            maxgain = -1
            splitter = None
            for Att in A:
                #print('A: ',A)
                #print('Att: ',Att)
                i = self.info_gain(df, Att, Attributes, Method)
                #if Method = 'EN':
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
                    child = self.build_tree(df.loc[idx],remaining_cols, Attributes, Label, current_depth+1,method)
                    children.append(child)
                
            return Node(splitter,len(Svi),maxgain,None,children)
    
    #predict value of a given row recursively
    def predict(self,node,row):
        if node.value != None:
            #print(node.value)
            return node.value
        
        #datapoint value for attribute that his node decides on
        value = row[node.attributesplit]
        attribute_vals = self.Attributes[node.attributesplit]
        
        
        for i, child in enumerate(node.children):
            #rint('Attributes[node.attributesplit][i]: ',Attributes[node.attributesplit][i])
            #print('value: ',value)
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
            


        
#gini_index(df)
mytree = Tree((maxdepth), Attributes, Label)
a = mytree.build_tree(df, A, Attributes, Label, 0, method)

error = mytree.calc_error(a,df)
print('trainingerror = ',error)