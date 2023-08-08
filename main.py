import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('MNIST-csv/train.csv')

# Separate the labels from the features
labels = data['label']
images = data.drop('label', axis=1)

labels = np.array(labels)
images = np.array(images)

# Shuffling/Randomizing
indices = np.arange(labels.shape[0])
np.random.shuffle(indices)
labels = labels[indices]
images = images[indices]

# Normalizing images
images = images / 255.0

# Making the labels also follow one-hot encoding to allow for cost function analysis.
n_classes = 10
Y = np.eye(n_classes)[labels]
Y_test = Y[0:10000].T

# Forward Propagation Helpers

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    cache = Z 
    return A, cache

def softmax(Z):

    """
    Implement the softmax function. This will be used in the output layer.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    
    cache = Z
    
    return A, cache

def initialize_params(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def linear_forward(L_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    L_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "L_prev", "W" and "b" ; stored for computing the backward pass efficiently
    """
    Z = np.dot(W, L_prev) + b
    
    cache = (L_prev, W, b)
    
    return Z, cache

def forward_prop_hidden(L_prev, W, b):
    Z, linear_cache = linear_forward(L_prev, W, b)
    A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_prop_output(L_prev, W, b):
    Z, linear_cache = linear_forward(L_prev, W, b)
    A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def compute_loss(A, Y):
    """
    Implement the cross-entropy loss function.

    Arguments:
    A -- Probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- True "label" vector (one-hot encoded), shape (1, number of examples)

    Returns:
    loss -- Cross-entropy loss
    """

    m = Y.shape[1]
    loss = -1/m * np.sum(Y * np.log(A))
    return loss

def accuracy(A, Y):
    """
    Calculate accuracy of the predicted labels.

    Arguments:
    A -- Predicted labels, shape (1, number of examples)
    Y -- True labels, one-hot encoded, shape (1, number of examples)

    Returns:
    accuracy -- Ratio of correctly predicted observations to the total observations
    """
    
    predictions = np.argmax(A, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy


# Back propagation Helpers

def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def linear_backward(dZ, cache):
    L_prev, W, b = cache
    m = L_prev.shape[1]

    dW = 1./m * np.dot(dZ, L_prev.T)
    db = 1./m * np.sum(dZ, axis=1, keepdims=True)
    dL_prev = np.dot(W.T, dZ)

    return dL_prev, dW, db

def softmax_backward(Y, cache):
    Z = cache
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    dZ = A - Y
    return dZ

# Main Code for Forward, Backward Propogation and Updating Params.

def forward(X, Y, parameters):
    A1, cache_1 = forward_prop_hidden(X, parameters['W1'], parameters['b1'])
    A2, cache_2 = forward_prop_output(A1, parameters['W2'], parameters['b2'])
    return A2, cache_1, cache_2, parameters

def backward_prop(A2, Y_test, cache1, cache2):
    linear_cache1, activation_cache1 = cache1
    linear_cache2, activation_cache2 = cache2

    # Backpropagation for the softmax activation function
    dZ2 = softmax_backward(Y_test, activation_cache2)
    dA1, dW2, db2 = linear_backward(dZ2, linear_cache2)

    # Backpropagation for the ReLU activation function
    dZ1 = relu_backward(dA1, activation_cache1)
    _, dW1, db1 = linear_backward(dZ1, linear_cache1)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def update_params(parameters, grads, alpha):
    parameters["W1"] = parameters["W1"] - alpha * grads["dW1"]
    parameters["b1"] = parameters["b1"] - alpha * grads["db1"]
    parameters["W2"] = parameters["W2"] - alpha * grads["dW2"]
    parameters["b2"] = parameters["b2"] - alpha * grads["db2"]
    return parameters

def gradient_descent(X, Y, alpha, iterations):
    layer_dims = [784, 50, 10]
    parameters = initialize_params(layer_dims)
    
    for i in range(iterations):
        A2, cache_1, cache_2, parameters = forward(X, Y, parameters)
        loss = compute_loss(A2, Y)
        grads = backward_prop(A2, Y, cache_1, cache_2)
        parameters = update_params(parameters, grads, alpha)
        
        if i % 10 == 0:
            acc = accuracy(A2, Y)
            print("Iteration: ", i, " Loss: ", loss, " Accuracy: ", acc)

    return parameters


def main():
    alpha = 0.1
    iterations = 600
    X = images[0:10000].T
    Y = Y_test
    
    # Performing gradient descent
    parameters = gradient_descent(X, Y, alpha, iterations)

    return parameters

parameters = main()