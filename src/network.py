"""
network
~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a
feedforward neural network. Gradients are calculated using backpropagation.
The activation function of each neuron is the sigmoid, and the cost function is
the cross-entropy. Also, L2 regularization is applied to reduce overfitting.
"""


import random
import numpy as np


def _sigmoid(z_l):
    return 1.0 / (1.0 + np.exp(-z_l))

class SigmoidActivation():

    @staticmethod
    def fn(z_l):
        """
        Returns the activation `a_l` of a layer of neurons given the output `z_l`
        of that layer of neurons.
        """

        return _sigmoid(z_l)

    @staticmethod
    def derivative(z_l):
        """
        Returns the derivative of the activation function, given the output `z_l`
        of a layer of neurons. In this particular case, we actually return a vector,
        since the jacobian for the sigmoid activation would be a diagonal matrix.
        """

        return _sigmoid(z_l) * (1.0 - _sigmoid(z_l))

class CrossEntropyCost():

    @staticmethod
    def fn(a_L, y):
        """
        Return the cost associated with an output `a_L` and desired output `y`.
        Note that np.nan_to_num is used to ensure numerical stability. In particular,
        0.0 * log(0.0) = 0.0 * -Inf = NaN, when the error should actually be 0.0.
        """

        return np.sum(np.nan_to_num(-y*np.log(a_L)-(1-y)*np.log(1-a_L)))

    @staticmethod
    def delta_L(z_L, a_L, y):
        """
        Return the error `delta_L` from the output layer.  Note that the parameter
        `z_L` is not used by the method.  It is included in the method's parameters
        in order to make the interface consistent with the delta method for other
        cost classes.
        """

        return a_L - y


class Network:

    def __init__(self, sizes):
        """
        The list `sizes` contains the number of neurons in the respective layers
        of the network. For example, if the list was [2, 3, 1] then it would be
        a three-layer network, with the first layer containing 2 neurons, the
        second layer 3 neurons, and the third layer 1 neuron. The biases and
        weights  for the network are initialized randomly, using
        `self.init_weights_and_biases`.
        """

        self.rng = np.random.default_rng()

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.init_weights_and_biases()

        self.activation = SigmoidActivation
        self.cost = CrossEntropyCost

    def init_weights_and_biases(self):
        """
        Initialize each weight using a Gaussian distribution with mean 0 and
        standard deviation 1 over the square root of the number of weights connecting
        to the same neuron. Initialize the biases using a Gaussian distribution
        with mean 0 and standard deviation 1.

        Note that the first layer is assumed to be an input layer, and by convention
        we won't set any biases for those neurons, since biases are only ever used
        in computing the outputs from later layers.
        """

        self.weights = [
            self.rng.standard_normal((y, x)) / np.sqrt(x)
                for x, y in zip(self.sizes[0:], self.sizes[1:])
        ]
        self.biases = [
            self.rng.standard_normal((y, 1))
                for y in self.sizes[1:]
        ]

    def SGD(
        self,
        training_data,
        epochs,
        mini_batch_size,
        eta,
        lmbda,
        validation_data=None,
        monitor_training_accuracy=False,
        monitor_validation_accuracy=False,
    ):
        """
        Train the neural network using mini-batch stochastic gradient descent.
        `training_data` is a list of tuples `(x, y)` representing the training
        inputs and the desired outputs. The other non-optional parameters are
        self-explanatory.

        The method also accepts `validation_data`. We can monitor the cost and
        accuracy on either the training data or validation data, by setting the
        appropriate flags.
        """

        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(
                    mini_batch,
                    eta,
                    lmbda,
                    n
                )

            print(f'Epoch {j} training complete')

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data)
                print(f'(training data) Accuracy: {accuracy} / {n}')

            if validation_data is not None:
                n_validation = len(validation_data)
                if monitor_validation_accuracy:
                    accuracy = self.accuracy(validation_data)
                    print(f'(validation data) Accuracy: {accuracy} / {n_validation}')

            print()

    def update_mini_batch(self, mini_batch, eta, lmbda, n):
        """
        Update the network's weights and biases by appling gradient descent
        using backpropagation to a single mini-batch. `mini_batch` is a list of
        tuples `(x, y)`, `eta` is the learning rate, `lmbda` is the regularization
        parameter, and `n` is the total size of the training dataset.
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
            (1.0 - eta * lmbda / n) * w - (eta / m) * np.sum(nwCx, axis=0)
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
            a = self.activation.fn(z)

            z_list.append(z)
            a_list.append(a)

        # backward pass
        delta = \
            self.cost.delta_L(z_list[-1], a_list[-1], y)

        nabla_w_C_x[-1] = delta @ a_list[-2].swapaxes(-2, -1)
        nabla_b_C_x[-1] = delta

        # Note that the variable `l` in the loop below is used a little differently
        # to the notation in Chapter 2 of the book. Here, `l = 1` means the last
        # layer of neurons, `l = 2` is the second-last layer, and so on. It's a
        # renumbering of the scheme in the book, used here to take advantage of
        # the fact that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            # for now we are expecting `activation.derivative` to be a vector and
            # not a jacobian matrix, that is, the activation function should be
            # local (one output should only depend on one input, so that the jacobian
            # turns out to be a diagonal matrix)
            delta = \
                self.activation.derivative(z_list[-l]) * self.weights[-l+1].transpose() @ delta

            nabla_w_C_x[-l] = delta @ a_list[-l-1].swapaxes(-2, -1)
            nabla_b_C_x[-l] = delta

        return (nabla_w_C_x, nabla_b_C_x)

    def accuracy(self, data):
        """
        Return the number of inputs in `data` for which the neural network outputs
        the correct result. The neural network's output is assumed to be the index
        of whichever neuron in the final layer has the highest activation.
        """

        results = [
            (np.argmax(self.feedforward(x)), y)
                for x, y in data
        ]
        return sum(
            y[x][0] == 1.0 for x, y in results
        )

    def feedforward(self, x):
        """
        Return the output of the network if `x` is input.
        """

        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.activation.fn(w @ a + b)

        return a
