import tensorflow as tf 
import numpy as np

# Define the critic loss for the first network using the Wasserstein cost fuction
# Define the critic loss for the first network using the Wasserstein cost fuction
def first_critic_loss(hr_tiles, sr_tiles, lr_tiles,  first_critic_model):
    # Calculate Gradient penalty
    Lambda = 10.0
    BATCH_SIZE = sr_tiles.shape[0]
    alpha = tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0.,maxval=1)
    differences = sr_tiles - hr_tiles
    interpol = hr_tiles + alpha*differences
    gradients = tf.gradients(first_critic_model((interpol, lr_tiles)), [interpol])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    #return tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))) - tf.reduce_mean(first_critic_model((hr_tiles, lr_tiles))) + Lambda*gradient_penalty
    return (tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))), - tf.reduce_mean(first_critic_model((hr_tiles, lr_tiles))), Lambda*gradient_penalty,
    tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))) - tf.reduce_mean(first_critic_model((hr_tiles, lr_tiles))) + Lambda*gradient_penalty)

def first_generator_loss(hr_tiles, sr_tiles, lr_tiles, first_critic_model):
    difference = sr_tiles-hr_tiles
    # Lambda_l1 = 0.5
    # pre_l1 = tf.norm(difference, ord=1, axis=1)
    # l1 = tf.norm(pre_l1, ord=1, axis=1)
    # l1 = Lambda_l1*tf.reduce_mean(l1)

    Lambda_inf = 0.
    pre_l_inf = tf.norm(difference, ord=np.inf, axis=1)
    l_inf = tf.norm(pre_l_inf, ord=np.inf, axis=1)
    l_inf = Lambda_inf*tf.reduce_mean(l_inf)

    return (-tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))), l_inf, -tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))) + l_inf)
    # return (-tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))), l1, -tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))) + l1)

# Define the critic loss for the second network using the Wasserstein cost fuction
def second_critic_loss(hr_tiles, sr_tiles, lr_tiles, second_critic_model):
    # Calculate Gradient penalty
    Lambda = 10.0
    BATCH_SIZE = sr_tiles.shape[0]
    alpha = tf.random.uniform(shape=[BATCH_SIZE, 1, 1, 1], minval=0.,maxval=1)
    differences = sr_tiles - hr_tiles
    interpol = hr_tiles + alpha*differences
    gradients = tf.gradients(second_critic_model((interpol, lr_tiles)), [interpol])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)

    return tf.reduce_mean(second_critic_model((sr_tiles, lr_tiles))) - tf.reduce_mean(second_critic_model((hr_tiles, lr_tiles))) + Lambda*gradient_penalty

def second_generator_loss(hr_tiles, sr_tiles, lr_tiles, second_critic_model):
    Lambda_l1 = 0.5
    difference = sr_tiles-hr_tiles
    pre_l1 = tf.norm(difference, ord=1, axis=1)
    l1 = tf.norm(pre_l1, ord=1, axis=1)
    l1 = Lambda_l1*tf.reduce_mean(l1)

    return -tf.reduce_mean(second_critic_model((sr_tiles, lr_tiles))) + l1