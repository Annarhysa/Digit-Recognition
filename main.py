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
x_train = x[:60000, :]
y_train = y[:60000]

x_test = x[60000:, :]
y_test = y[60000:]

m = x.shape[0]
#Images are of (28x28) px so there will be 784 features
input_layer_size = 784
hidden_layer_size = 100
num_labels = 10

#randomly initialising thetas
it1 = initialise(hidden_layer_size, input_layer_size)
it2 = initialise(num_labels, hidden_layer_size)

#parameters in a single column
i_nn_params = np.concatenate((it1.flatten(), it2.flatten()))
max_iter = 100
lambda_reg = 0.1 # to avoid overfitting
args = (input_layer_size, hidden_layer_size, num_labels, x_train, y_train, lambda_reg)

#Calling minimize function to reduce cost function and to train weights
results = minimize()
