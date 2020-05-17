import tensorflow as tf 
import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers

# Define the model for the generator based on the desired sr factor


def first_res_block_mult(input_batch, block, features1, features2):
    X = layers.Conv2D((features1), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer1')(input_batch)
    X = layers.LayerNormalization()(X)
    X = layers.ReLU()(X)

    X = layers.Conv2D((features2), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer2')(X)
    X = layers.LayerNormalization()(X)

    X_shortcut = layers.Conv2D((features2), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_shortcut')(input_batch)

    X = tf.add(X, X_shortcut)
    X = layers.ReLU()(X)
    return X

def first_generator(input_batch, factor=2):

    X = first_res_block_mult(input_batch, 1, 4, 8)
    X = first_res_block_mult(X, 2, 16, 32)
    X = first_res_block_mult(X, 3, 32, 32)

    X = layers.UpSampling2D()(X)

    X = first_res_block_mult(X, 4, 32, 16)
    X = first_res_block_mult(X, 5, 8, 4)
    X = layers.Conv2D((1), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'gen_final')(X)
    X = layers.ReLU()(X)

    #X_short = tf.image.resize(input_batch, size=(input_batch.shape[1]*factor, input_batch.shape[1]*factor), method=tf.image.ResizeMethod.BICUBIC)
    #X = tf.add(X, tf.dtypes.cast(X_short, dtype='float64'))
    return X

def first_critic(input_batch, factor=2):
    X = layers.Conv2D((8), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit1')(input_batch)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((16), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit2')(X)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit3')(X)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit4')(X)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.AveragePooling2D()(X)

    X = layers.Conv2D((32), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit5')(X)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((8), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit6')(X)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((2), (3, 3), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit7')(X)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)    
    
    X = layers.Flatten()(X)
    X = layers.Dense(1)(X)
    return X

def second_res_block_mult(input_batch, block, features1, features2):
    X = layers.Conv2D((features1), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer1')(input_batch)
    #X = layers.LayerNormalization()(X)
    X = layers.ReLU()(X)

    X = layers.Conv2D((features2), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_layer2')(X)
    #X = layers.LayerNormalization()(X)

    X_shortcut = layers.Conv2D((features2), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'block'+str(block)+'_shortcut')(input_batch)

    X = tf.add(X, X_shortcut)
    X = layers.ReLU()(X)
    return X


def second_generator(input_batch, factor=2):
    X = second_res_block_mult(input_batch, 1, 4, 8)
    X = second_res_block_mult(X, 2, 16, 32)

    X = second_res_block_mult(X, 3, 32, 16)
    X = second_res_block_mult(X, 4, 8, 4)
    X = layers.Conv2D((1), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'gen_final')(X)
    X = layers.ReLU()(X)

    #X_short = tf.image.resize(input_batch, size=(input_batch.shape[1]*factor, input_batch.shape[1]*factor), method=tf.image.ResizeMethod.BICUBIC)
    #X = tf.add(X, tf.dtypes.cast(X_short, dtype='float64'))
    return X

def second_critic(input_batch, factor=2):
    X = layers.Conv2D((8), (1, 1), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit1')(input_batch)
    X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((16), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit2')(X)
    #X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit3')(X)
    #X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit5')(X)
    #X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((32), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit6')(X)
    #X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((8), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit7')(X)
    #X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)

    X = layers.Conv2D((1), (5, 5), padding = 'same', kernel_initializer = keras.initializers.RandomNormal(), name = 'factor2crit8')(X)
    #X = layers.LayerNormalization()(X)
    X = layers.LeakyReLU()(X)    
    
    X = layers.Flatten()(X)
    X = layers.Dense(1)(X)
    return X
