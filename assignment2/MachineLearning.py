# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:51:19 2018

@author: Felix Farias Fueyo 
"""
from numpy import exp, random,dot, sqrt, log, array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#Forecast one output variable for a set of predictors.
class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.synaptic_weights = 2*random.random((number_of_inputs_per_neuron,number_of_neurons))

class NeuralNetwork:
    def __init__(self, layer1, layer2, activation_function):
        self.layer1 = layer1
        self.layer2 = layer2
        self.activation_function = activation_function
        
     #Define activation functions 
    def __Sigmoid(self, x,deriv=False):
        if(deriv==True):
             return x*(1-x)
            
        return 1/(1+exp(-x))
    
    def __ReLu(self, x,deriv=False):
        if(deriv==True):
            return 1/(1+exp(-x))
            
        return log(1+exp(x))
    
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

            layer2_error = training_set_outputs - output_from_layer_2 
            print( "Error:" + str(np.mean(np.abs(layer2_error))))
            if self.activation_function == 'ReLu':
                layer2_delta = np.multiply(layer2_error, self.__ReLu(output_from_layer_2, deriv = True))
            elif self.activation_function == 'Sigmoid':
                layer2_delta = np.multiply(layer2_error, self.__Sigmoid(output_from_layer_2, deriv = True))
                  
            layer1_error = layer2_delta.dot(self.layer2.synaptic_weights.T)
            if self.activation_function == 'ReLu':
                layer1_delta = np.multiply(layer1_error, self.__ReLu(output_from_layer_1, deriv = True))
            elif self.activation_function == 'Sigmoid':
                layer1_delta = np.multiply(layer1_error, self.__Sigmoid(output_from_layer_1, deriv = True))
                        
            layer1_adjustment = training_set_inputs.T.dot(layer1_delta)
            layer2_adjustment = output_from_layer_1.T.dot(layer2_delta)
            
            #When using ReLu it is important to restrict the learning rate, such that the 
            #gradient does not blow-up the optimization.
            self.layer1.synaptic_weights += 0.001*layer1_adjustment
            self.layer2.synaptic_weights += 0.001*layer2_adjustment
    
    def think(self, inputs):
        if self.activation_function == 'ReLu':
            output_from_layer_1 = self.__ReLu(dot(inputs, self.layer1.synaptic_weights))
            output_from_layer_2 = self.__ReLu(dot(output_from_layer_1, self.layer2.synaptic_weights))
        elif self.activation_function == 'Sigmoid':
            output_from_layer_1 = self.__Sigmoid(dot(inputs, self.layer1.synaptic_weights))
            output_from_layer_2 = self.__Sigmoid(dot(output_from_layer_1, self.layer2.synaptic_weights))
        
        return output_from_layer_1, output_from_layer_2
    
    def print_weights(self):
        print( ' Layer 1 : ')
        print( self.layer1.synaptic_weights)
        print( ' Layer 2 : ')
        print( self.layer2.synaptic_weights)
            
        
    def rmse(self, output, test_set):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_squared_error
        return sqrt(mean_squared_error(test_set, output))
    
    def mad(self, output, test_set):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(test_set, output)
    
    #test_set should be the realizations of the dependent variable, output should be the estimated values
    def ndcg(self, result):
        all_ndcg = pd.DataFrame(result['srch_id'].unique())
        all_ndcg.columns = ['srch_id']
        
        all_ndcg = all_ndcg.set_index('srch_id')
        result = result.set_index('srch_id')
        all_ndcg['ndcg'] = math.nan
        all_ndcg['dcg'] = math.nan
        
        result['rel'] = result['booking_bool']*5 + result['click_bool'] #should have 5 as maximum
        result.loc[result['rel'] == 6, 'rel'] = 5 # 6 can only occure if click and bool are 1. Hence, cap at 5.
        result = result.drop(['booking_bool', 'click_bool'], axis = 1)
        
        it = 0
        for i in result.index.unique().values:
            rel_temp = result.loc[i]  #Extract the search query for which the accuracy will be evaluated
            rel_temp = rel_temp.sort_values('pred', ascending = False) #Sort the query based on the estimated probabilities
            
            irel = rel_temp.sort_values('rel', ascending = False) #compute ideal scenario (highest DCG score)
            
            IDCG = 0
            DCG = 0 #Compute discounted cumulative gain
            for j in range(1,len(rel_temp)+1):
                DCG = DCG + (2**rel_temp.iloc[j-1].rel-1)/np.log2(j + 1)
                IDCG = IDCG + (2**irel.iloc[j-1].rel-1)/np.log2(j + 1)
            
            all_ndcg.loc[i, 'ndcg'] = DCG/IDCG  #Compute normalized discounted cumulative gain. (within [0,1] range)
            all_ndcg.loc[i, 'dcg'] = DCG
            
            it = it + 1
            print([it/len(result.index.unique()), DCG/IDCG])
            
        return np.mean(all_ndcg)