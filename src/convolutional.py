# %%
import tensorflow as tf
from tensorflow import keras


# %%
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1)) / 255
x_test = x_test.reshape((10000, 28, 28, 1)) / 255


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
    x=x_train,
    y=y_train,
    epochs=60,
    batch_size=10,
    validation_split=0.2,
)

