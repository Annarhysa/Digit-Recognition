#importing all the necesaary libraries
from scipy.io import loadmat
import numpy as np
from model import neural_network
from randinitialise import initialise
from prediction import  predict
from scipy.optimize import minimize

#loading the dataset
data = loadmat('data/mnist.mat')

#extracting features and labels from the file
x = data['data']
y = data['label']
x = x.transpose()
y = y.flatten()

#normalizing the data
x = x/255

#splitting data into training and testing

