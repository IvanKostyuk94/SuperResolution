import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

def first_res_block_mult(input_batch, block, features1, features2):
    X = layers.Conv2D((features1), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer1')(input_batch)
    X = layers.ReLU()(X)

    X = layers.Conv2D((features2), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer2')(X)

    X_shortcut = layers.Conv2D((features2), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_shortcut')(input_batch)

    X = tf.add(X, X_shortcut)
    X = layers.ReLU()(X)
    return X

def generator(input_batch):
    X_short = input_batch
    X = first_res_block_mult(input_batch, 1, 4, 8)
    X = first_res_block_mult(X, 2, 16, 32)
    X = first_res_block_mult(X, 3, 32, 16)
    X = first_res_block_mult(X, 4, 8, 4)
    
    X = layers.Conv2D((2), (1, 1), padding = 'same', use_bias=True, kernel_initializer = keras.initializers.RandomNormal(), name = 'gen_final')(X)
    X = X_short+X
    return X