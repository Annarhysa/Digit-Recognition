import numpy as np

#creating a function generate random numbers (weights)

def initialise(a,b):
    epsilon = 0.15
    c = np.random.rand(a,b+1) * (2*epsilon) - epsilon   #generates weights between [-epsilon, +epsilon]
    return c