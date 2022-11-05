import tensorflow as tf 

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
    return tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles))) - tf.reduce_mean(first_critic_model((hr_tiles, lr_tiles))) + Lambda*gradient_penalty

def first_generator_loss(hr_tiles, sr_tiles, lr_tiles, first_critic_model):
    return -tf.reduce_mean(first_critic_model((sr_tiles, lr_tiles)))

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
    return -tf.reduce_mean(second_critic_model((sr_tiles, lr_tiles)))
