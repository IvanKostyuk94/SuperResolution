# Define the training steps for thetraining procedure 

import tensorflow as tf 

tf.random.set_seed(42)


# Define the training step using the tf function to compile it (first network)
@tf.function
def first_train_step(input_tiles, labels, first_generator_model, first_critic_model, 
                    first_generator_loss, first_critic_loss, first_generator_optimizer, first_critic_optimizer):
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as crit_tape:
        sr_tiles = first_generator_model(input_tiles)
    
        gen_loss = first_generator_loss(labels, sr_tiles, input_tiles, first_critic_model)
        crit_loss = first_critic_loss(labels, sr_tiles, input_tiles, first_critic_model)

    gradients_of_generator = gen_tape.gradient(gen_loss, first_generator_model.trainable_variables)
    gradients_of_critic = crit_tape.gradient(crit_loss, first_critic_model.trainable_variables)
   
    first_generator_optimizer.apply_gradients(zip(gradients_of_generator, first_generator_model.trainable_variables))
    first_critic_optimizer.apply_gradients(zip(gradients_of_critic, first_critic_model.trainable_variables))

    return (gen_loss, crit_loss, gradients_of_generator, gradients_of_critic)



# Define the training step for only the critic using the tf function to compile it (first network)
@tf.function
def first_train_step_crit(input_tiles, labels, first_generator_model, first_critic_model, first_critic_loss, first_critic_optimizer):

    with tf.GradientTape() as crit_tape:
        sr_tiles = first_generator_model(input_tiles)
        crit_loss = first_critic_loss(labels, sr_tiles, input_tiles, first_critic_model)

    gradients_of_critic = crit_tape.gradient(crit_loss, first_critic_model.trainable_variables)

    first_critic_optimizer.apply_gradients(zip(gradients_of_critic, first_critic_model.trainable_variables))
    
    return (crit_loss, gradients_of_critic)
