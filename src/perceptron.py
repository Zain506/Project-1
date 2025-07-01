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
    - predict(): Predict output of a given array
    - train(): Train using dataframe, specify ground truth column, batch size and learning rate pre-set and customisable
    - evaluate(): Test perceptron and obtain error on dataset. Specify ground truth column
    """
    def __init__(self, params: int, function: str) -> None: # Add reshaping inputs from ndArrays to 1d
        self.W = np.ones(params) * 0.1 # Initial setting of W by number of parameters
        self.b = np.ones(1) * 0.1
        # Velocities for momentum-based gradient descent
        self.vW = 0
        self.vb = 0
        
        activations = {
            "sigmoid": sigmoid,
            "relu": lambda x: np.where(x > 0, x, 0.05*x),
            "tanh": lambda x: math.tanh(x)
        }
        derivatives = {
            "sigmoid": lambda x: sigmoid(x)*(1-sigmoid(x)),
            "relu": lambda x: np.where(x > 0, 1, -0.05),
            "tanh": lambda x: 1 - np.tanh(x)*np.tanh(x)
        }

        if function not in activations or function not in derivatives:
            self.error = "Invalid Activation Function"

        self.activation = activations.get(function)
        self.derivative = derivatives.get(function)

    def predict(self, x: np.array):
        # Complete forward pass
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

            dLoss = 2*(pred - y_batch) # Derivative of loss
            dAct = self.derivative(pred) # Derivative of wTX+b
            deriv = dLoss*dAct # Chain Rule
            
            # Gradient Descent
            db = deriv.mean()
            dW = np.dot(x_batch.T, deriv) / batch_size # Vector
            self._gradDescent(dW, db, lr = lr)
    
    def evaluate(self, df: pd.DataFrame, gt: str) -> float: 
        # Obtain error
        y = df[gt]
        x = df.drop(columns=[gt])
        pred = self._forward(x)
        vect = (y-pred)*(y-pred)
        return vect.mean()

    def _forward(self, inputs: pd.DataFrame) -> np.ndarray: 
        # Forward pass with given inputs
        agg = (np.dot(inputs, self.W) - self.b)
        out = self.activation(agg)
        return (out)
    
    def _gradDescent(self, dW: np.array, db: np.array, lr) -> None:
        # Vanilla Gradient Descent
        self.W -= lr * dW
        self.b -= lr * db
    
    def _momentumGradDescent(self, dW: np.array, db: np.array, lr, m = 0.9) -> None:
        # Momentum Gradient Descent
        self.vW = m*self.vW - lr*dW
        self.vb = m*self.vb - lr*db
        self.W += self.vW
        self.b += self.vb