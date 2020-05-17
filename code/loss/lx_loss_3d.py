import tensorflow as tf 
import numpy as np

def first_generator_loss(hr_tiles, sr_tiles):
    difference = sr_tiles-hr_tiles
    
    Lambda_l1 = 0.5
    #Lambda_l1 = 0.0
    l1 = tf.norm(difference, ord=1, axis=1) #axis 1
    l1 = tf.norm(l1, ord=1, axis=1) #axis 2
    l1 = tf.norm(l1, ord=1, axis=1) #axis 3
    l1 = Lambda_l1*tf.reduce_mean(l1)

    #Lambda_l2 = 10.
    Lambda_l2 = 0.
    l2 = tf.norm(difference, ord=2, axis=1)
    l2 = tf.norm(l2, ord=2, axis=1)
    l2 = tf.norm(l2, ord=2, axis=1)
    l2 = Lambda_l2*tf.reduce_mean(l2)

    #Lambda_l_inf = 50.
    Lambda_l_inf = 0.
    l_inf = tf.norm(difference, ord=np.inf, axis=1)
    l_inf = tf.norm(l_inf, ord=np.inf, axis=1)
    l_inf = tf.norm(l_inf, ord=np.inf, axis=1)
    l_inf = Lambda_l_inf*tf.reduce_mean(l_inf)

    l_tot = l1+l2+l_inf
    return (l1, l2, l_inf, l_tot)