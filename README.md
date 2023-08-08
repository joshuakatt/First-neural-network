# First-neural-network
 Creating my first, and very own neural network from the ground up on the MNIST dataset. It's a Multilayer Perceptron (MLP) neural network from scratch in Python, trained it on the MNIST dataset to recognize handwritten digits, and evaluated its performance. Built simply using Python and NumPy.

### Read my notes and Lessons I've learnt on this here: [Read Here](https://docs.google.com/document/d/1cApv3LuWQ7Qqe2DwDib_tNvy8w__2vC0Wcn-FEhqKUw/edit?usp=sharing)

## Skills & Concepts Applied
Deep understanding of how neural networks work
Data preprocessing and normalization
Working with numpy for matrix operations
Implementation of forward propagation and backpropagation
Applying the gradient descent algorithm
Evaluating a machine learning model's performance
Data Preprocessing
The MNIST dataset was preprocessed by performing the following steps:

## Splitting the data into features (images) and labels
Normalizing the pixel values to fall between 0 and 1
One-hot encoding the labels
The Neural Network Model
The model consists of an input layer, two hidden layers, and an output layer. Each layer employs an activation function. For the hidden layers, the Rectified Linear Unit (ReLU) function is used, while the output layer uses the Softmax function, making it suitable for multi-class classification.

## Training the Model
Training the model involved feeding the data through the network (forward propagation), calculating the loss (cross-entropy loss function), and then adjusting the weights and biases to minimize this loss. This last step, known as backpropagation, is achieved using the gradient descent algorithm.

## Evaluation
The model's performance was evaluated using the accuracy metric, which measures the proportion of true results (both true positives and true negatives) in the dataset.

## Future Improvements
This is a foundational project, and there are numerous avenues for extension:

Experimenting with different activation functions
Implementing regularization techniques to avoid overfitting
Making the neural network architecture more complex by adding more layers or neurons
Implementing other types of gradient descent, like mini-batch or stochastic<br>
Hope you find this project insightful. Feel free to contribute and make improvements!
