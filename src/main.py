import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data()


import network

net = network.Network([784, 30, 10])
net.SGD(
    training_data=training_data,
    validation_data=validation_data,
    epochs=30,
    mini_batch_size=10,
    eta=0.005,
    mu=0.99,
    lmbda=5.0,
    monitor_training_cost=True,
    monitor_validation_cost=True,
    monitor_training_accuracy=True,
    monitor_validation_accuracy=True,
)

net = network.Network([784, 100, 10])
net.SGD(
    training_data=training_data,
    validation_data=validation_data,
    epochs=60,
    mini_batch_size=10,
    eta=0.001,
    mu=0.99,
    lmbda=5.0,
    monitor_training_cost=True,
    monitor_validation_cost=True,
    monitor_training_accuracy=True,
    monitor_validation_accuracy=True,
)
