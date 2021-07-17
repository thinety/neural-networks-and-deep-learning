"""
network
~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a
feedforward neural network. Gradients are calculated using backpropagation.
"""


import random
import numpy as np


def sigmoid(z):
    """
    The sigmoid function.
    """

    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    """
    Derivative of the sigmoid function.
    """

    return sigmoid(z) * (1 - sigmoid(z))


class Network:

    def __init__(self, sizes):
        """
        The list `sizes` contains the number of neurons in the respective layers
        of the network. For example, if the list was [2, 3, 1] then it would be
        a three-layer network, with the first layer containing 2 neurons, the
        second layer 3 neurons, and the third layer 1 neuron. The biases and weights
        for the network are initalized randomly, using a Gaussian distribution with
        mean 0 and variance 1. Note that the first layer is assumed to be an input
        layer, and by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later layers.
        """

        self.rng = np.random.default_rng()

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [
            self.rng.standard_normal((y, x))
                for x, y in zip(sizes[0:], sizes[1:])
        ]
        self.biases = [
            self.rng.standard_normal((y, 1))
                for y in sizes[1:]
        ]

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        `training_data` is a list of tuples `(x, y)` representing the training
        inputs and the desired outputs. The other non-optional parameters are
        self-explanatory. If `test_data` is provided then the network will be
        evaluated agains the test data after each epoch, and partial progress
        printed out. This is useful for tracking progress, but slows things down
        substantially.
        """

        n = len(training_data)
        if test_data is not None:
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data is not None:
                print(f'Epoch {j}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {j} complete')

    def update_mini_batch(self, mini_batch, eta):
        """
        Update the network's weights and biases by appling gradient descent
        using backpropagation to a single mini-batch. `mini_batch` is a list of
        tuples `(x, y)` and `eta` is the learning rate.
        """

        nabla_C_w = [
            np.zeros(w.shape)
                for w in self.weights
        ]
        nabla_C_b = [
            np.zeros(b.shape)
                for b in self.biases
        ]

        for x, y in mini_batch:
            delta_nabla_C_w, delta_nabla_C_b = self.backprop(x, y)
            nabla_C_w = [
                nCw + dnCw
                    for nCw, dnCw in zip(nabla_C_w, delta_nabla_C_w)
            ]
            nabla_C_b = [
                nb + dnb
                    for nb, dnb in zip(nabla_C_b, delta_nabla_C_b)
            ]

        self.weights = [
            w - (eta / len(mini_batch)) * nCw
                for w, nCw in zip(self.weights, nabla_C_w)
        ]
        self.biases = [
            b - (eta / len(mini_batch)) * nCb
                for b, nCb in zip(self.biases, nabla_C_b)
        ]

    def backprop(self, x, y):
        """
        Return a tuple `(nabla_C_w, nabla_C_b)` representing the gradient of the
        cost function C_x. `nabla_C_w` and `nabla_C_b` are layer-by-layer lists
        of numpy arrays, similar to `self.biases` and `self.weights`.
        """

        pass

    def cost_derivative(self, output_activations, y):
        """
        Return the vector of partial derivatives \partial C_x / \partial a for
        the output activations.
        """

        pass

    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural network outputs the
        correct result. Note that the neural network's output is assumed to be the
        index of whichever neuron in the final layer has the highest activation.
        """

        test_results = [
            (np.argmax(self.feedforward(x)), y)
                for x, y in test_data
        ]
        return sum(
            y[x][0] for x, y in test_results
        )

    def feedforward(self, a):
        """
        Return the output of the network if `a` is input.
        """

        for w, b in zip(self.weights, self.biases):
            a = sigmoid(w @ a + b)

        return a
