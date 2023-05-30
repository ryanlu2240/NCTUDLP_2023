import numpy as np
from scipy import signal


# Base class
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        raise NotImplementedError

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        raise NotImplementedError



# inherit from base class Layer
class FCLayer(Layer):
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size, optimizer):
        self.weights = np.random.rand(output_size, input_size) #- 0.5
        self.bias = np.random.rand(output_size, 1) #- 0.5
        self.optimizer = optimizer
        self.delta_prev = 0
        self.alpha = 0.1

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.weights, self.input) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_error, learning_rate):
        
        input_error = np.dot(self.weights.T, output_error)
        weights_error = np.dot(output_error, self.input.T)
        bias_error = output_error
        if self.optimizer=='sgd':
            # update parameters
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * bias_error
        if self.optimizer=='momentum':
            # update parameters
            delta = learning_rate * weights_error + self.alpha * self.delta_prev
            self.delta_prev = delta
            self.weights -= delta
            self.bias -= learning_rate * bias_error
        
        return input_error
    

# inherit from base class Layer
class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    # returns the activated input
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    # Returns input_error=dE/dX for a given output_error=dE/dY.
    # learning_rate is not used because there is no "learnable" parameters.
    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input) * output_error
    
class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size): # output shape = (input_height - kernel_size + 1, input_width - kernel_size + 1)
        input_height, input_width = input_shape # w * h
        self.kernels_shape = (kernel_size, kernel_size)
        self.input_shape = input_shape
        self.kernels = np.random.randn(kernel_size, kernel_size)
        self.biases = np.random.randn(input_height - kernel_size + 1, input_height - kernel_size + 1)

    def forward_propagation(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        self.output += signal.correlate2d(self.input, self.kernels, "valid")
        return self.output

    def backward_propagation(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        kernels_gradient = signal.correlate2d(self.input, output_gradient, "valid")
        input_gradient += signal.convolve2d(output_gradient, self.kernels, "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_propagation(self, input):
        return np.reshape(input, self.output_shape)

    def backward_propagation(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)
