"""
Author: Zain Nomani
Date: 29 June 2025
Description: Single Layer Perceptron with loss function
"""
import numpy as np
import pandas as pd
import math
def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1+np.exp(-x))

class Perceptron:
    """
    Single Layer Perceptron
        - Parameters W and b
        - Multiple Activation functions
        - Gradient Descent capabilities (require step size selection)
    """
    def __init__(self, params: int, function: str) -> None: # Add reshaping inputs from ndArrays to 1d
        self.W = np.ones(params) # Initial setting of W by number of parameters
        self.b = np.ones(1)
        self.error = None # Error parameter to check if problem with initialisation
        activations = {
            "sigmoid": sigmoid,
            "relu": lambda x: max(0, x),
            "tanh": lambda x: math.tanh(x)
        }
        derivatives = {
            "sigmoid": lambda x: sigmoid(x)*(1-sigmoid(x)),
            "relu": lambda x: int(x>0),
            "tanh": lambda x: 1 - np.tanh(x)*np.tanh(x)
        }
        if function not in activations or function not in derivatives:
            self.error = "Invalid Activation Function"
        self.activation = activations.get(function)
        self.derivative = derivatives.get(function)

    def predict(self, x: np.array):
        return (self._forward(x))

    def train(self, df: pd.DataFrame, gt: str, batch_size = 256, lr = 10) -> None:
        # Default to stochastic gradient descent
        N = df.shape[0]

        for i in range(0, N, batch_size):
            y = df[gt].values
            x = df.drop(columns=[gt]).values
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            pred = self._forward(x_batch) # Forward pass

            dLoss = 2*(y_batch - pred) # Derivative of loss
            dAct = pred*(1-pred) # Derivative of wTX+b
            deriv = dLoss*dAct # Chain Rule
            
            # Gradient Descent
            db = deriv.mean()
            dW = np.dot(x_batch.T, deriv) / batch_size # Vector
            self._gradDescent(dW, db, lr = lr)
    def evaluate(self, df: pd.DataFrame, gt: str) -> float: # Obtain error
        y = df[gt]
        x = df.drop(columns=[gt])
        pred = self._forward(x)
        vect = (y-pred)*(y-pred)
        return vect.mean()

    def _forward(self, inputs: pd.DataFrame) -> np.ndarray: # Forward pass with given inputs
        agg = (np.dot(inputs, self.W) - self.b)
        out = self.activation(agg)
        return (out)
    
    def _gradDescent(self, dW: np.array, db: np.array, lr) -> None: # Backpropagation of W and b
        self.W -= lr * dW
        self.b -= lr * db