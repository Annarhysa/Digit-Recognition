import numpy as np

def neural_network(params, input_size, hidden_size, labels, x, y, lamb):
    #splitting weights into theta1 and theta2
    theta1 = np.reshape(params[:hidden_size*(input_size+1)],
                        (hidden_size, input_size + 1))
    theta2 = np.reshape(params[hidden_size*(input_size+1):],
                        (labels, hidden_size + 1))
    
    m = x.shape[0]
    # Forward propagation: compute a1 and z2
    matrix = np.ones((m,1))
    #adding bias to first layer
    x = np.append(matrix, x, axis =1) 
    a1 = x
    z2 = np.dot(x, theta1.transpose())
    #activation function for 2nd layer
    a2 = 1/(1+np.exp(-z2))
    matrix = np.ones((m,1))
    #adding bias to layer 2
    a2 = np.append(matrix, a2, axis = 1)
    z3 = np.dot(a2, theta2.transpose())
    a3 = 1/(1+np.exp(-z3))

    #y values to vector
    y_vect = np.zeros((m,10))
    for i in range(m):
        y_vect[i, int(y[i])] = 1

    #Calculate the cost function
    J = (1 / m) * (np.sum(np.sum(-y_vect * np.log(a3) - (1 - y_vect) * np.log(1 - a3)))) + (lamb / (2 * m)) * (
                sum(sum(pow(theta1[:, 1:], 2))) + sum(sum(pow(theta2[:, 1:], 2))))

    #Backpropagation
    Delta3 = a3 - y_vect
    Delta2 = np.dot(Delta3, theta2) * a2 * (1 - a2)
    Delta2 = Delta2[:, 1:]

    #Gradient descent
    theta1[:, 0] = 0
    Theta1_grad = (1 / m) * np.dot(Delta2.transpose(), a1) + (lamb / m) * theta1
    theta2[:, 0] = 0
    Theta2_grad = (1 / m) * np.dot(Delta3.transpose(), a2) + (lamb / m) * theta2
    grad = np.concatenate((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return J,grad