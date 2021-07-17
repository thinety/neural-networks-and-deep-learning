import numpy as np


import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def cost_derivative(a_L, y):
    return a_L - y


import network

net = network.Network([784, 30, 10], sigmoid, sigmoid_prime, cost_derivative)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

net = network.Network([784, 100, 10], sigmoid, sigmoid_prime, cost_derivative)
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
