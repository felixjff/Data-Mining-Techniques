# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:51:19 2018

@author: Felix Farias Fueyo 
"""
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

#Forecast one output variable for a set of predictors.
class NeuralNetwork:
    'Defines Neural Network'
    
    def __init__(self, dVariable, train_set, hidden_layers, neurons, activation_function):
        'Initiate variables'
        
        self.train = train_set #training set with dependent and indipendent variables (data frame)
        self.neurons = neurons #neurons in hidden layers. Input as vector, entry one indicates
        #neurons for first hidden layer, entriy 2 for second, etc.
        self.hidden_layers = hidden_layers
        self.dVariable = dVariable #Dependent variable
        self.activation_function = activation_function #name of activation function to be used
        
    def check(self):
        print(self.train)
        
    def train_network1h(self):
        #extract number of input neurons from the data frame
        in_neurons = self.train.columns.size - 1
        out_values = len(self.train)
        #extract dependent variable
        y = self.train[self.dVariable]
        
        #Define activation functions 
        def Sigmoid(x,deriv=False):
            if(deriv==True):
                return x*(1-x)
            
            return 1/(1+np.exp(-x))
        def ReLu(x,deriv=False):
            if(deriv==True):
                return 1/(1+np.exp(-x))
            
            return np.log(1+np.exp(x))
        
        np.random.seed(1)
        
        #Define weights for a network with one hidden layer and one output variable
        # randomly initialize our weights with mean 0.
        syn0 = np.random.random((in_neurons,out_values)) - 1
        syn1 = np.random.random((out_values,1)) - 1
        syn1 = syn1.flatten()
        
        l0 = self.train.drop(self.dVariable, axis = 1)
        #Train network
        for j in range(10000):
            # Feed forward through layers 0, 1, and 2
            if self.activation_function == 'Sigmoid':
                l1 = Sigmoid(np.dot(l0,syn0))
                l2 = Sigmoid(np.dot(l1,syn1))
            elif self.activation_function == 'ReLu':
                l1 = ReLu(np.dot(l0,syn0))
                l2 = ReLu(np.dot(l1,syn1))
   
            l2 = l2.flatten()
            #Compute error at output layer
            l2_error = y - l2
            #Performance tracker...
            if (j% 100) == 0:
                print("Error:" + str(np.mean(np.abs(l2_error))))
            #Compute weight correction term for output synampses
            if self.activation_function == 'Sigmoid':
                l2_delta = l2_error*Sigmoid(l2,deriv=True)
            elif self.activation_function == 'ReLu':
                l2_delta = l2_error*ReLu(l2,deriv=True)
            #Compute error at hidden layer
            l1_error = l2_delta.dot(syn1.T.flatten())
            #Compute weight correction term for synampses to first hidden layer
            if self.activation_function == 'Sigmoid':
                l1_delta = l1_error*Sigmoid(l1,deriv=True)
            elif self.activation_function == 'ReLu':
                l1_delta = l1_error * ReLu(l1,deriv=True)
                
            #Update weights
            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
        #Return weights to be used in forcasting
        return [syn1, syn0]
    
    def train_network(self):
        #extract number of input neurons from the data frame
        in_neurons = self.train.columns.size - 1
        out_values = len(self.train)
        #extract dependent variable
        y = self.train[self.dVariable]
        
        #Define activation functions 
        def Sigmoid(x,deriv=False):
            if(deriv==True):
                return x*(1-x)
            
            return 1/(1+np.exp(-x))
        def ReLu(x,deriv=False):
            if(deriv==True):
                return 1/(1+np.exp(-x))
            
            return np.log(1+np.exp(x))
        
        np.random.seed(1)
        
        #Define weights for a network with one hidden layer and one output variable
        # randomly initialize our weights with mean 0.
        syn0 = np.random.random((in_neurons,out_values)) - 1
        syn1 = np.random.random((out_values,1)) - 1
        syn1 = syn1.flatten()
        
        l0 = self.train.drop(self.dVariable, axis = 1)
        
        for j in range(60000):
                        # Feed forward through layers 0, 1, and 2
            if self.activation_function == 'Sigmoid':
                l1 = Sigmoid(np.dot(l0,syn0))
                l2 = Sigmoid(np.dot(l1,syn1))
            elif self.activation_function == 'ReLu':
                l1 = ReLu(np.dot(l0,syn0))
                l2 = ReLu(np.dot(l1,syn1))
            
            l2_delta = (y - l2)*(l2*(1-l2))
            l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
            
            if (j% 100) == 0:
                print("Error:" + str(np.mean(np.abs(y - l2))))
            
            syn1 += l1.T.dot(l2_delta)
            syn0 += l0.T.dot(l1_delta)
        
        return [syn1, syn0]
    
    #def predict(self, weights):
        
    def rmse(self, prediction, test_set):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(test_set,prediction))