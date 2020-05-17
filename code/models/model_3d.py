''' A single pass 3D network with 3x3x3 convolutional kernels'''

import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

# Define the model for the generator based on the desired sr factor


def first_res_block_mult(input_batch, block, features1, features2):
    X = layers.Conv3D((features1), (3, 3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block{}_layer1'.format(block))(input_batch)
    X = layers.ReLU()(X)

    X = layers.Conv3D((features2), (3, 3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block{}_layer2'.format(block))(X)

    X_shortcut = layers.Conv3D((features2), (1, 1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block{}_shortcut'.format(block))(input_batch)

    X = tf.add(X, X_shortcut)
    X = layers.ReLU()(X)
    return X

def first_generator(input_batch):
    X_short = input_batch
    X = first_res_block_mult(input_batch, 1, 2, 4)
    X = first_res_block_mult(X, 2, 8, 16)
    X = first_res_block_mult(X, 3, 32, 16)
    X = first_res_block_mult(X, 4, 8, 4)
    
    X = layers.Conv3D((1), (1, 1, 1), padding = 'same', use_bias=False, kernel_initializer = keras.initializers.RandomNormal(), name = 'gen_final')(X)
    X = X + X_short
    return X

def first_critic(input_batch, lr_batch):
    X = input_batch #-lr_batch
    #X = input_batch
    # X = layers.Conv2D((8), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit1')(X)
    # X = layers.LeakyReLU()(X)

    X = layers.Conv2D((16), (5, 5), bias_initializer = keras.initializers.RandomNormal() , padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit2')(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (5, 5), bias_initializer = keras.initializers.RandomNormal() , padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit3')(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (5, 5), bias_initializer = keras.initializers.RandomNormal() , padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit4')(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((8), (5, 5), bias_initializer = keras.initializers.RandomNormal() , padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit5')(X)
    X = layers.LeakyReLU()(X) 

    X = layers.Conv2D((2), (5, 5), bias_initializer = keras.initializers.RandomNormal() , padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'crit6')(X)
    X = layers.LeakyReLU()(X)    
    
    X = layers.Flatten()(X)
    X = layers.Dense(1, use_bias=False)(X)
    return X
