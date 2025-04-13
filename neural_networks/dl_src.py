import numpy as np

class Linear():

    '''
    This class contains the methods required for the Linear layer in a neural network.
    '''

    def __init__(self, n_output_nodes: int, bias: bool,
                 batch_size: int, activation: str):
        self.n_output_nodes = n_output_nodes
        self.bias = bias
        self.batch_size = batch_size
        self.weights = None
        self.activation = activation
        self.activation_fxns = Activations()

    def initialize_weights(self, n_input_features: int):
        '''
        Remember that the first column of the generated weights is bias
        '''
        self.weights = np.random.random_sample(size=(self.n_output_nodes, n_input_features))-0.3

    def linear_operation(self, input_data):
        '''
        This function does the W_i(.)X_i linear operation
        '''
        if self.weights is None:
            self.initialize_weights(n_input_features=input_data.shape[1])

        if self.bias:
            bias_input = np.random.rand(self.batch_size, 1)
            bias_weights = np.random.rand(self.weights.shape[0], 1)
            input_data = np.hstack((bias_input, input_data))
            self.weights = np.hstack((bias_weights, self.weights))

        if input_data.shape[0] != self.batch_size:
            input_data = input_data[:self.batch_size]

        output = input_data.dot(self.weights.T)

        if self.activation == 'relu':
            return self.activation_fxns.relu(output)

        if self.activation == 'sigmoid':
            return self.activation_fxns.sigmoid(output)


class Activations:

    '''
    This class contains all the activations used in deep learning
    '''

    def __init__(self):
        pass

    def relu(self, input_data):
        '''
        Passes the input through Rectified Linear Unit.
        f(x) = {
                x : x > 0,
                0 : x < 0
                }
        '''
        return np.maximum(0, input_data)

    def sigmoid(self, input_data):
        '''
        Passes the input through sigmoid function.
        The input is expected as W_i(.)X_i
        OUTPUT : 0 <= f(x) <= 1
        '''
        return 1/(1-np.exp(input_data))
