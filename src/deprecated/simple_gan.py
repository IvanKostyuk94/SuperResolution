import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Define the model for the generator based on the desired sr factor


def first_generator(input_batch):
    X_short = input_batch

    X = layers.Conv2D(
        (4),
        (5, 5),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="layer1",
    )(input_batch)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (8),
        (5, 5),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="layer2",
    )(input_batch)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (1),
        (1, 1),
        padding="same",
        use_bias=True,
        kernel_initializer=keras.initializers.RandomNormal(),
        name="final_layer",
    )(X)
    X = layers.ReLU()(X)
    X = X  # + X_short
    return X


def first_critic(input_batch, lr_batch):
    X = input_batch  # -lr_batch
    # X = input_batch
    # X = layers.Conv2D((8), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit1')(X)
    # X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (16),
        (5, 5),
        bias_initializer=keras.initializers.RandomNormal(),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="crit2",
    )(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (32),
        (5, 5),
        bias_initializer=keras.initializers.RandomNormal(),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="crit3",
    )(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (32),
        (5, 5),
        bias_initializer=keras.initializers.RandomNormal(),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="crit4",
    )(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (8),
        (5, 5),
        bias_initializer=keras.initializers.RandomNormal(),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="crit5",
    )(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D(
        (2),
        (5, 5),
        bias_initializer=keras.initializers.RandomNormal(),
        padding="same",
        kernel_initializer=keras.initializers.RandomNormal(),
        name="crit6",
    )(X)
    X = layers.LeakyReLU()(X)

    X = layers.Flatten()(X)
    X = layers.Dense(1, use_bias=False)(X)
    return X
