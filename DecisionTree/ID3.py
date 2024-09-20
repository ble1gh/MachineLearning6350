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
maxdepth = 2

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
            
    #find indeces of all rows in main dataframe given an attribute and value, return
    #a list of index sets for each value the attribute can take, and a new list of 
    #attributes with that one removed
    def split(self,df,col,cols,A):
        Sv = []
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
            val = df.label.mode().to_list()
            return Node(value=val)
        else:
            maxgain = float(0)
            splitter = None
            for Att in A:
                print('A: ',A)
                #print('Att: ',Att)
                i = self.info_gain(df, Att, Attributes, Method)
                print('i: ',i)
                if i > maxgain:
                    maxgain = i
                    splitter = Att
            #print(splitter)
            Svi, remaining_cols = self.split(df,splitter,A,Attributes)
            #print('Svi: ',Svi)
            
            children = []
            for i in Svi:
                #print(df.loc[i])
                child = self.build_tree(df.loc[i],remaining_cols, Attributes, Label, current_depth+1,method)
                children.append(child)
                
            return Node(splitter,len(Svi),maxgain,None,children)

# Function to count the total number of leaf nodes in the tree
def count_leaf_nodes(node):
    if node.value is not None:  # It's a leaf node
        return 1
    return sum(count_leaf_nodes(child) for child in node.children)

# Function to plot the decision tree
def plot_tree(node, x=0.5, y=1.0, x_offset=1, y_offset=0.1, ax=None, parent_pos=None):
    if ax is None:
        # Initialize the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_axis_off()

    # Plot current node
    if node.value is not None:
        # Leaf node
        label = f'Leaf: {node.value}'
    else:
        # Decision node
        label = f'{node.attributesplit}\nInfo Gain: {round(node.infogain, 3)}'

    ax.text(x, y, label, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black'))

    if parent_pos:
        # Draw an edge from the parent node to the current node
        ax.plot([parent_pos[0], x], [parent_pos[1], y], 'k-', lw=1)

    if node.children:
        num_leaf_nodes = count_leaf_nodes(node)  # Count total leaf nodes under this node
        step = x_offset / (num_leaf_nodes - 1) if num_leaf_nodes > 1 else 0

        child_x = x - x_offset / 2  # Start the children on the left
        for idx, child in enumerate(node.children):
            # Recursively plot children
            new_x = child_x + step * count_leaf_nodes(child) / 2
            plot_tree(child, x=new_x, y=y - y_offset, x_offset=x_offset / 2, y_offset=y_offset, ax=ax, parent_pos=(x, y))
            child_x += step * count_leaf_nodes(child)

    return ax
        
mytree = Tree((maxdepth), Attributes, Label)
a = mytree.build_tree(df, A, Attributes, Label, 0, method)
# Example usage with your tree structure
ax = plot_tree(a)  # 'a' is the root node of the tree created by build_tree()

# Display the tree
plt.show()
#a = a.to_list()
#print(a.value)
#Sv, remaining_cols = split(df,cols[5],A,Attributes)
#info_gain(df,cols[5],Attributes['safety'],'EN')
#e = df.loc[df[cols[5]] == 'low']