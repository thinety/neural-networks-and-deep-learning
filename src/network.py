"""
network
~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a
feedforward neural network. Gradients are calculated using backpropagation.
"""


import random
import numpy as np


class Network:

    def __init__(self, sizes, activation, activation_prime, cost_derivative):
        """
        The list `sizes` contains the number of neurons in the respective layers
        of the network. For example, if the list was [2, 3, 1] then it would be
        a three-layer network, with the first layer containing 2 neurons, the
        second layer 3 neurons, and the third layer 1 neuron. The biases and weights
        for the network are initalized randomly, using a Gaussian distribution with
        mean 0 and variance 1. Note that the first layer is assumed to be an input
        layer, and by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later layers.

        `activation` a function used as activation function for the neurons.
        `activation_prime` is its derivative. `cost_derivative` is a function that
        receives the output of the network and the expected output, and returns
        `nabla_{a^L} C_x`, that is, the vector of partial derivatives
        `\partial C_x / \partial a^L_j`, where `C_x` is the cost function for a
        single training sample, and `a^L_j` are the
        output layer activations.
        """

        self.rng = np.random.default_rng()

        self.num_layers = len(sizes)
        self.weights = [
            self.rng.standard_normal((y, x))
                for x, y in zip(sizes[0:], sizes[1:])
        ]
        self.biases = [
            self.rng.standard_normal((y, 1))
                for y in sizes[1:]
        ]

        self.activation = activation
        self.activation_prime = activation_prime
        self.cost_derivative = cost_derivative

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

        x = np.array([
            x for x, _ in mini_batch
        ])
        y = np.array([
            y for _, y in mini_batch
        ])
        nabla_w_C_x, nabla_b_C_x = self.backprop(x, y)

        m = len(mini_batch)
        self.weights = [
            w - (eta / m) * np.sum(nwCx, axis=0)
                for w, nwCx in zip(self.weights, nabla_w_C_x)
        ]
        self.biases = [
            b - (eta / m) * np.sum(nbCx, axis=0)
                for b, nbCx in zip(self.biases, nabla_b_C_x)
        ]

    def backprop(self, x, y):
        """
        Return a tuple `(nabla_{w} C_x, nabla_{b} C_x)` representing the gradient of the
        cost function C_x. `nabla_{w} C_x` and `nabla_{b} C_x` are layer-by-layer lists
        of numpy arrays, similar to `self.biases` and `self.weights`.
        """

        nabla_w_C_x = [
            np.zeros(w.shape)
                for w in self.weights
        ]
        nabla_b_C_x = [
            np.zeros(b.shape)
                for b in self.biases
        ]

        # feedforward
        a = x

        z_list = []  # list to store all z vectors, layer by layer
        a_list = [a] # list to store all a vectors, layer by layer

        for w, b in zip(self.weights, self.biases):
            z = w @ a + b
            a = self.activation(z)

            z_list.append(z)
            a_list.append(a)

        # backward pass
        delta = \
            self.cost_derivative(a_list[-1], y) * self.activation_prime(z_list[-1])

        nabla_w_C_x[-1] = delta @ a_list[-2].swapaxes(-2, -1)
        nabla_b_C_x[-1] = delta

        # Note that the variable `l` in the loop below is used a little differently
        # to the notation in Chapter 2 of the book. Here, `l = 1` means the last
        # layer of neurons, `l = 2` is the second-last layer, and so on. It's a
        # renumbering of the scheme in the book, used here to take advantage of
        # the fact that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            delta = \
                self.weights[-l+1].transpose() @ delta * self.activation_prime(z_list[-l])

            nabla_w_C_x[-l] = delta @ a_list[-l-1].swapaxes(-2, -1)
            nabla_b_C_x[-l] = delta

        return (nabla_w_C_x, nabla_b_C_x)

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

    def feedforward(self, x):
        """
        Return the output of the network if `x` is input.
        """

        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.activation(w @ a + b)

        return a
