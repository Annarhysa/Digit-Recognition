import numpy as np

def predict(theta1, theta2, x):
    m = x.shape[0]
    matrix = np.ones((m,1))

    #adding bias to the first layer
    x = np.append(matrix, x, axis = 1)
    z2 = np.dot(x, theta1.transpose())

    #activation for the second layer
    a2 = 1/(1+np.exp(-z2))
    matrix = np.ones((m,1))

    #adding bias to second layer
    a2 = np.append(matrix, a2,axis=1)
    z3 = np.dot(a2, theta2.transpose())

    #activation of third layer
    a3 = 1/(1+np.exp(-z3))
    p = (np.argmax(a3, axis = 1)) #prediciting the class

    return p
