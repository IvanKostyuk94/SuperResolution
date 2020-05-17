# Define the training steps for thetraining procedure 

import tensorflow as tf 

# Define the training step using the tf function to compile it (first network)
@tf.function
def first_train_step(input_tiles, labels, 
                    gen_cost_l1, gen_cost_l2,
                    gradient_gen_l1, gradient_gen_l2, 
                    first_generator_model, first_generator_loss, 
                    first_generator_optimizer):
    with tf.GradientTape() as gen_tape_l1:
        sr_tiles = first_generator_model(input_tiles)
        gen_loss_l2 = first_generator_loss(labels, sr_tiles)
        gen_cost_l2 = gen_loss_l2 + gen_cost_l2


    l2_gradients_of_generator = gen_tape_l1.gradient(gen_loss_l2, first_generator_model.trainable_variables)

    #Save the total value of the gradients l1
    for variable in l2_gradients_of_generator:
        tot_grad_gen_l2 = tf.sqrt(tf.reduce_sum(tf.square(variable)))
        gradient_gen_l2 = tf.add(tot_grad_gen_l2, gradient_gen_l2) 

    first_generator_optimizer.apply_gradients(zip(l2_gradients_of_generator, first_generator_model.trainable_variables))
    return (gen_cost_l1, gen_cost_l2, gradient_gen_l1, gradient_gen_l2, sr_tiles)


# Define the training step using the tf function to compile it (second network)
@tf.function
def second_train_step(input_tiles, labels, 
                    gen_cost_l1, gen_cost_l2,
                    gradient_gen_l1, gradient_gen_l2, 
                    second_generator_model, second_generator_loss,
                    second_generator_optimizer):
    with tf.GradientTape() as gen_tape_l1:
        sr_tiles = second_generator_model(input_tiles)
        gen_loss_l2 = second_generator_loss(labels, sr_tiles)
        gen_cost_l2 = gen_loss_l2 + gen_cost_l2
    
    l2_gradients_of_generator = gen_tape_l1.gradient(gen_loss_l2, second_generator_model.trainable_variables)

    #Save the total value of the gradients l1
    for variable in l2_gradients_of_generator:
        tot_grad_gen_l2 = tf.sqrt(tf.reduce_sum(tf.square(variable)))
        gradient_gen_l2 = tf.add(tot_grad_gen_l2, gradient_gen_l2) 
   
    second_generator_optimizer.apply_gradients(zip(l2_gradients_of_generator, second_generator_model.trainable_variables))
    return (gen_cost_l1, gen_cost_l2, gradient_gen_l1, gradient_gen_l2)