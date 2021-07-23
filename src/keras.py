# %%
import tensorflow as tf
from tensorflow import keras


# %%
def load_mnist_data():
    import gzip
    import numpy as np

    with gzip.open('../data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = f.read()[16:]
        train_images = np.frombuffer(train_images, np.uint8).reshape((60000, 784))

        x_train = train_images / 255

    with gzip.open('../data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = f.read()[8:]
        train_labels = np.frombuffer(train_labels, np.uint8).reshape((60000,))

        y_train = np.zeros((60000, 10), np.float64)
        for j, label in enumerate(train_labels):
            y_train[j][label] = 1.0

    with gzip.open('../data/test-images-idx3-ubyte.gz', 'rb') as f:
        test_images = f.read()[16:]
        test_images = np.frombuffer(test_images, np.uint8).reshape((10000, 784))

        x_test = test_images / 255

    with gzip.open('../data/test-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = f.read()[8:]
        test_labels = np.frombuffer(test_labels, np.uint8).reshape((10000,))

        y_test = np.zeros((10000, 10), np.float64)
        for j, label in enumerate(test_labels):
            y_test[j][label] = 1.0

    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_mnist_data()


# %%
def _l2(l2, w):
    return l2 * 0.5 * tf.reduce_sum(tf.square(w))
class L2:
    def __init__(self, l2):
        self.l2 = l2
    def __call__(self, w):
        return _l2(self.l2, w)

def _nan_to_zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
def _categorical_crossentropy(y_true, y_pred):
    return -tf.reduce_sum(
        _nan_to_zero(y_true * tf.math.log(y_pred) + (1-y_true) * tf.math.log(1-y_pred)),
        axis=-1,
    )
class CategoricalCrossentropy:
    def __call__(self, y_true, y_pred):
        return _categorical_crossentropy(y_true, y_pred)

def _categorical_accuracy(y_true, y_pred):
    return tf.cast(
        tf.equal(
            tf.argmax(y_true, axis=-1),
            tf.argmax(y_pred, axis=-1),
        ),
        tf.float32,
    )
class CategoricalAccuracy:
    def __call__(self, y_true, y_pred):
        return _categorical_accuracy(y_true, y_pred)


# %%
inputs = keras.Input(shape=(784,))

x = keras.layers.Dense(
    units=100,
    activation=tf.nn.sigmoid,
    kernel_regularizer=L2(0.0001),
)(inputs)

outputs = keras.layers.Dense(
    units=10,
    activation=tf.nn.sigmoid,
    kernel_regularizer=L2(0.0001),
)(x)

model = keras.Model(inputs=inputs, outputs=outputs)


# %%
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=0.001,
        momentum=0.99,
    ),
    loss=CategoricalCrossentropy(),
    metrics=[
        CategoricalAccuracy(),
    ],
)

model.fit(
    x=x_train,
    y=y_train,
    epochs=60,
    batch_size=10,
    validation_split=0.2,
)

