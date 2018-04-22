# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 19:51:19 2018

@author: Felix Farias Fueyo 
"""
from numpy import exp, random,dot, sqrt, log, array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            self.layer1.synaptic_weights += 0.00003*layer1_adjustment
            self.layer2.synaptic_weights += 0.00003*layer2_adjustment
    
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
            
    
    #def predict(self, weights):
        
    def rmse(self, output, test_set):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_squared_error
        return sqrt(mean_squared_error(test_set, output))
    def mad(self, output, test_set):
        'use self.dVariable and test_set'
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(test_set, output)