import numpy as np
import tensorflow as tf
import keras
from keras import layers
from Evaluation import evaluation


# Function to create a primary capsule layer
def primary_capsule_layer(inputs, num_capsules, capsule_dim, kernel_size, strides, padding):
    conv2d = layers.Conv2D(num_capsules * capsule_dim,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding=padding)(inputs)
    outputs = tf.reshape(conv2d, [-1, num_capsules, capsule_dim])
    outputs = squash(outputs)
    return outputs


# Squashing activation function
def squash(vectors):
    norm = tf.norm(vectors, axis=-1, keepdims=True)
    return (norm ** 2 / (1 + norm ** 2)) * (vectors / (norm + 1e-9))


# Function to create an adaptive routing capsule layer
def adaptive_routing_capsule_layer(inputs, num_capsules, capsule_dim, num_routing_iters=3):
    routing_weights = layers.Dense(num_capsules * capsule_dim)(inputs)
    routing_weights = tf.reshape(routing_weights, [-1, 1, num_capsules, capsule_dim])

    for i in range(num_routing_iters):
        coupling_coefficients = tf.nn.softmax(routing_weights, axis=2)
        weighted_predictions = tf.reduce_sum(coupling_coefficients * inputs, axis=2, keepdims=True)
        outputs = squash(weighted_predictions)

        if i < num_routing_iters - 1:
            agreement = tf.reduce_sum(inputs * outputs, axis=-1, keepdims=True)
            routing_weights += agreement

    return tf.squeeze(outputs, axis=2)


# Function to build the Adaptive Capsule Network model
def build_adaptive_capsnet(sol):
    num_classes = 1
    input_shape = (28, 28, 1)
    inputs = keras.Input(shape=input_shape)

    # First convolutional layer
    x = layers.Conv2D(256, kernel_size=9, strides=1, padding='valid', activation='relu')(inputs)

    # Primary capsule layer
    x = primary_capsule_layer(x, num_capsules=32, capsule_dim=8, kernel_size=9, strides=2, padding='valid')

    # Adaptive routing capsule layer
    x = adaptive_routing_capsule_layer(x, num_capsules=num_classes, capsule_dim=16)

    # Flatten capsules to pass to dense layer
    x = layers.Flatten()(x)

    # Fully connected dense layer for classification
    outputs = layers.Dense(sol[0], num_classes, activation='softmax')(x)

    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=outputs, name='adaptive_capsnet')
    return model


def Model_ACNet(train_data, Train_Target, test_data, Test_Target, sol=None):
    if sol is None:
        sol = [5, 5, 300]
    IMG_SIZE = 28
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    # Create an instance of the Adaptive Capsule Network model
    model = build_adaptive_capsnet(sol)
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(Train_X, Train_Target, epochs=sol[1], batch_size=64, steps_per_epoch=sol[2], verbose=2)
    Pred = model.predict(Test_Target)
    Eval = evaluation(Pred, Test_Target)
    return Eval, Pred
