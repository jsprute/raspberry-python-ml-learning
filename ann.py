import numpy as np

class ANN:

    def __init__(self, inode, hnode, onode, lr):
        # Set local variables
        self.inode = inode
        self.hnode = hnode
        self.onode = onode
        self.lr = lr
        
        # Mean is the reciprocal of the sqrt of node sum
        mean = 1 / (pow((inode + hnode + onode), 0.5))

        #Std dev is approx 1/6 of total weight range
        # total range = 2
        sd = 2 / 6

        # Generate both weight matrices
        # Input to hiddent layer    
        self.wtgih = np.random.normal(mean, sd, [hnode, inode])

        # Hidden to output layer
        self.wtgho = np.random.normal(mean, sd, [onode, hnode])



    def testNet(self, input):
        # Convert input data vector into an array
        input = np.array(input, ndmin=2).T

        # Multiple input array by wtgih matrix
        hInput = np.dot(self.wtgih, input)

        # Apply activation function
        hOutput = 1 / (1 + np.exp(-hInput))

        # Multiply hidden layer output by wtgho matrix
        oInput = np.dot(self.wtgho, hOutput)

        # Apply activation function
        oOutput = 1 / (1 + np.exp(-oInput))
        
        return oOutput

    def trainNet(self, inputT, train):
        # This module depends on values, arrays, and matrices
        # created when the init module is run
        # Create the arrays from the list arguments
        self.inputT = np.array(inputT, ndmin=2).T
        self.train = np.array(train, ndmin=2).T


