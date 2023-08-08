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

def forward_prop_hidden(L_prev, W, b, activation):
    Z, linear_cache = linear_forward(L_prev, W, b)
    A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def forward_prop_output(L_prev, W, b, activation):
    Z, linear_cache = linear_forward(L_prev, W, b)
    A, activation_cache = softmax(Z)
    cache = (linear_cache, activation_cache)
    return A, cache


def forward():
    