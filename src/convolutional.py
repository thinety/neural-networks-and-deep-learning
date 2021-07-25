# %%
import numpy as np
import tensorflow as tf
from tensorflow import keras


# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1)) / 255
x_test = x_test.reshape((10000, 28, 28, 1)) / 255

n_train = x_train.shape[0]
x_train_expanded = np.zeros((5*n_train, 28, 28, 1), np.float64)
y_train_expanded = np.zeros((5*n_train,), np.uint8)
for i in range(n_train):
    x_train_expanded[5*i] = x_train[i]
    x_train_expanded[5*i+1, :-1, :] = x_train[i, 1:, :]
    x_train_expanded[5*i+2, 1:, :] = x_train[i, :-1, :]
    x_train_expanded[5*i+3, :, :-1] = x_train[i, :, 1:]
    x_train_expanded[5*i+4, :, 1:] = x_train[i, :, :-1]

    y_train_expanded[5*i:5*i+5] = y_train[i]

n_test = x_test.shape[0]
x_test_expanded = np.zeros((5*n_test, 28, 28, 1), np.float64)
y_test_expanded = np.zeros((5*n_test,), np.uint8)
for i in range(n_test):
    x_test_expanded[5*i] = x_test[i]
    x_test_expanded[5*i+1, :-1, :] = x_test[i, 1:, :]
    x_test_expanded[5*i+2, 1:, :] = x_test[i, :-1, :]
    x_test_expanded[5*i+3, :, :-1] = x_test[i, :, 1:]
    x_test_expanded[5*i+4, :, 1:] = x_test[i, :, :-1]

    y_test_expanded[5*i:5*i+5] = y_test[i]


# %%
inputs = keras.Input(shape=(28, 28, 1))

x = keras.layers.Conv2D(
    filters=20,
    kernel_size=(5, 5),
    strides=(1, 1),
    activation=tf.nn.relu,
    kernel_regularizer=keras.regularizers.L2(1e-5),
)(inputs)
x = keras.layers.MaxPooling2D(
    pool_size=(2, 2),
)(x)

x = keras.layers.Conv2D(
    filters=40,
    kernel_size=(5, 5),
    strides=(1, 1),
    activation=tf.nn.relu,
    kernel_regularizer=keras.regularizers.L2(1e-5),
)(x)
x = keras.layers.MaxPooling2D(
    pool_size=(2, 2),
)(x)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(
    units=100,
    activation=tf.nn.relu,
    kernel_regularizer=keras.regularizers.L2(1e-5),
)(x)
outputs = keras.layers.Dense(
    units=10,
    activation=tf.nn.softmax,
    kernel_regularizer=keras.regularizers.L2(1e-5),
)(x)

model = keras.Model(inputs=inputs, outputs=outputs)


# %%
model.compile(
    optimizer=keras.optimizers.SGD(
        learning_rate=0.03,
    ),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(),
    ],
)

model.fit(
    x=x_train_expanded,
    y=y_train_expanded,
    epochs=60,
    batch_size=10,
    validation_split=0.2,
)

