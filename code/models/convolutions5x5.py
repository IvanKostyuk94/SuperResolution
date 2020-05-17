''' A multi pass network with the following number of parameters:
first generator =
first critic = 
second generator =
second critic =
It is designed to take as an input simulations put on a larger grid than appropriate without the need for upsampling. 
Thereby areas which are sufficiently sampled in the low-res simulation do not need to be unnecessarily upscaled again.
The proceeding goes as follows:

In the generator we save the input as the shortcut value and add it at the end to the network output to obtain the full picture. Hence the network only
needs to learn the residuals that have to be added to go from the lr_image to the hr_image without needing to transport all the information about the 
image intself

In the critic we subract the lr input from the input of the critic such that the critic only needs to examine the residual and can ignore the features that 
the lr and hr simulations have in common anyways.

Here we use a shortcut through the whole network thereby forcing the network to only learn the residuals thereby hopefully 
facilitating the learning process.

Furthermore since we work on the same scales there is no necessity to have two different network architectures.
'''

import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

# Define the model for the generator based on the desired sr factor


def first_res_block_mult(input_batch, block, features1, features2):
    X = layers.Conv2D((features1), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer1')(input_batch)
    X = layers.ReLU()(X)

    X = layers.Conv2D((features2), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer2')(X)

    X_shortcut = layers.Conv2D((features2), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_shortcut')(input_batch)

    X = tf.add(X, X_shortcut)
    X = layers.ReLU()(X)
    return X

def first_generator(input_batch):
    X_short = input_batch
    X = first_res_block_mult(input_batch, 1, 4, 8)
    X = first_res_block_mult(X, 2, 16, 32)
    X = first_res_block_mult(X, 3, 32, 16)
    X = first_res_block_mult(X, 4, 8, 4)
    
    X = layers.Conv2D((1), (1, 1), padding = 'same', use_bias=True, kernel_initializer = keras.initializers.RandomNormal(), name = 'gen_final')(X)
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

def second_res_block_mult(input_batch, block, features1, features2):
    X = layers.Conv2D((features1), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer1')(input_batch)
    X = layers.ReLU()(X)

    X = layers.Conv2D((features2), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer2')(X)

    X_shortcut = layers.Conv2D((features2), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_shortcut')(input_batch)

    X = tf.add(X, X_shortcut)
    X = layers.ReLU()(X)
    return X

def second_generator(input_batch):
    X_short =  input_batch
    X = second_res_block_mult(input_batch, 1, 4, 8)
    X = second_res_block_mult(X, 2, 16, 32)
    X = second_res_block_mult(X, 3, 32, 16)
    X = second_res_block_mult(X, 4, 8, 4)
    
    X = layers.Conv2D((1), (1, 1), padding = 'same', use_bias=True, kernel_initializer = keras.initializers.RandomNormal(), name = 'gen_final')(X)
    X = X #+ X_short
    return X

def second_critic(input_batch, lr_batch):
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
    X = layers.Dense(1, bias_initializer = keras.initializers.RandomNormal(), use_bias=False)(X)
    return X