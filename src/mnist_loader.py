import gzip
import numpy as np


def vectorized_result(j):
    """
    Return a (10, 1)-dimensional unit vector with a 1.0 in the jth position and
    zeroes elsewhere. This is used to convert a digit (0...9) into a corresponding
    desired output from the neural network.
    """

    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_data():
    """
    Return a tuple containing `(training_data, validation_data, test_data)`.

    In particular, `training_data` is a list containing 50'000 2-tuples `(x, y)`.
    `x` is a (784, 1)-dimensional numpy.ndarray containing the input image. `y` is a
    (10, 1)-dimensional numpy.ndarray representing the unit vector corresponding
    to the correct digit for `x`.

    `validation_data` and `test_data` are lists containing 10'000 items of the
    same format as `training_data`.
    """

    with gzip.open('../data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = f.read()
    with gzip.open('../data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = f.read()
    with gzip.open('../data/test-images-idx3-ubyte.gz', 'rb') as f:
        test_images = f.read()
    with gzip.open('../data/test-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = f.read()

    train_images = train_images[16:]
    training_inputs = [
        np.frombuffer(train_images[i:i+784], np.uint8).reshape((784, 1)) / 255
            for i in map(lambda x: 784*x, range(50000))
    ]
    validation_inputs = [
        np.frombuffer(train_images[i:i+784], np.uint8).reshape((784, 1)) / 255
            for i in map(lambda x: 784*x, range(50000, 60000))
    ]

    train_labels = train_labels[8:]
    training_results = [
        vectorized_result(train_labels[i])
            for i in range(50000)
    ]
    validation_results = [
        vectorized_result(train_labels[i])
            for i in range(50000, 60000)
    ]

    test_images = test_images[16:]
    test_inputs = [
        np.frombuffer(test_images[i:i+784], np.uint8).reshape((784, 1)) / 255
            for i in map(lambda x: 784*x, range(10000))
    ]

    test_labels = test_labels[8:]
    test_results = [
        vectorized_result(test_labels[i])
            for i in range(10000)
    ]

    training_data = list(zip(training_inputs, training_results))
    validation_data = list(zip(validation_inputs, validation_results))
    test_data = list(zip(test_inputs, test_results))

    return training_data, validation_data, test_data
