#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:17:35 2024

@author: brookleigh
"""

import numpy as np
import pandas as pd

#choose method of data splitting:
method = 'EN' #information gain (entropy)
#method = 'ME' #majority error
#method = 'GI'#gini-index

#maximum tree depth
maxdepth = 6

S = []
Label = ['unacc', 'acc', 'good', 'vgood']
A =[]
cols = []


#input data description
Attributes = {'age': 'numeric',
              'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
              'fnlwgt': 'numeric',
              'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
              'education-num': 'numeric',
              'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
              'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
              'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
              'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
              'sex': ['Female', 'Male'],
              'capital-gain': 'numeric',
              'capital-loss': 'numeric',
              'hours-per-week': 'numeric',
              'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                                 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos',
                                 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                                 'Holand-Netherlands']
    }


for Att in Attributes:
    cols.append(Att)
    
cols.append('label')
A = cols[0:-1]

    
df = pd.read_csv('train_final.csv',names = cols)
df_test = pd.read_csv('test_final.csv',names = A)
'''
#replace ? values with the most common known value in that category
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
dftest = pd.read_csv('bank/test.csv',names=cols)

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
'''

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
    
    
    def build_tree(self,df,A,Attributes,Label,current_depth,Method):
            
        if current_depth >= maxdepth or self.is_unique(df['label']) or A == []:
            val = df.label.mode()[0]
            return Node(value=val)
        else:
            maxgain = -1
            splitter = None
            #print(A)
            for Att in A:
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
            #print(node.value)
            return node.value
        
        #datapoint value for attribute that his node decides on
        value = row[node.attributesplit]
        attribute_vals = self.Attributes[node.attributesplit]
        
        
        for i, child in enumerate(node.children):
            #print('attribute_vals: ',attribute_vals)
            #print('value: ',value)
            if attribute_vals == 'numeric':
                if i == 0 and float(value)>node.divider:
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
        predictions = []
        
        # iterate over each row in the dataframe
        for idx, row in df.iterrows():
            #actual_label = row['label']
            #print('actual: ',actual_label)
            predicted_label = self.predict(tree, row)
            if predicted_label == None:
                predicted_label = 0
            predictions.append(predicted_label)
            print('prediction: ',predicted_label)

            #if predicted_label != actual_label:
                #incorrect_predictions += 1

        # Calculate and return the training error as a fraction of misclassified samples
        #training_error = incorrect_predictions / total_samples
        #print(incorrect_predictions)
        return (predictions)

        
#gini_index(df)
#avg_error = []
#for depth in range(maxdepth):
mytree = Tree(maxdepth, Attributes, Label)
a = mytree.build_tree(df, A, Attributes, Label, 0, method)
    
preds = mytree.calc_error(a,df_test)
#print('prediction error: ',error,' depth = ',maxdepth)
results = {'data':['test'], 'method': [method], 'maxdepth': [maxdepth]}
#labels = ['data', 'method', 'maxdepth','prediction error']

df_results = pd.DataFrame(data = preds)
print(df_results)

missing_val_count_by_column = (df_results.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])

df_results.to_csv('results_1.csv', mode='a', header=['Prediction'])
    #avg_error.append(error)
    #print('trainingerror = ',error,', depth = ',depth)
#avg = sum(avg_error)/len(avg_error)
#print('average error = ',avg)