# Define the training steps for thetraining procedure 

import tensorflow as tf 

# Define the training step using the tf function to compile it (first network)
@tf.function
def first_train_step(input_tiles, labels, 
                    gen_cost_l1, gen_cost_l2, gen_cost_l_inf,
                    gradient_gen_l1, gradient_gen_l2, gradient_gen_l_inf, 
                    first_generator_model, first_generator_loss, 
                    first_generator_optimizer, first_generator_optimizer_bias):
    
    with tf.GradientTape() as gen_tape_l1, tf.GradientTape() as gen_tape_l2, tf.GradientTape() as gen_tape_l_inf, tf.GradientTape() as full_tape:
        sr_tiles = first_generator_model(input_tiles)
        gen_loss_l1, gen_loss_l2, gen_loss_l_inf, loss_l1_l2 = first_generator_loss(labels, sr_tiles)

        gen_cost_l1 = tf.add(gen_loss_l1, gen_cost_l1)
        gen_cost_l2 = tf.add(gen_loss_l2, gen_cost_l2)
        gen_cost_l_inf = tf.add(gen_loss_l_inf, gen_cost_l_inf)

    l1_gradients_of_generator = gen_tape_l1.gradient(gen_loss_l1, first_generator_model.trainable_variables)
    l2_gradients_of_generator = gen_tape_l2.gradient(gen_loss_l2, first_generator_model.trainable_variables)
    l_inf_gradients_of_generator = gen_tape_l_inf.gradient(gen_loss_l_inf, first_generator_model.trainable_variables)

    l1_l2_gradients = full_tape.gradient(loss_l1_l2, first_generator_model.trainable_variables)

    #Save the total value of the gradients l1
    for variable in l1_gradients_of_generator:
        tot_grad_gen_l1 = tf.sqrt(tf.reduce_sum(tf.square(variable)))
        gradient_gen_l1 = tf.add(tot_grad_gen_l1, gradient_gen_l1) 

    #Save the total value of the gradients l2
    for variable in l2_gradients_of_generator:
        tot_grad_gen_l2 = tf.sqrt(tf.reduce_sum(tf.square(variable)))
        gradient_gen_l2 = tf.add(tot_grad_gen_l2, gradient_gen_l2)

    #Save the total value of the gradients l_inf
    for variable in l_inf_gradients_of_generator:
        tot_grad_gen_l_inf = tf.sqrt(tf.reduce_sum(tf.square(variable)))
        gradient_gen_l_inf = tf.add(tot_grad_gen_l_inf, gradient_gen_l_inf)

    first_generator_optimizer.apply_gradients(zip(l1_l2_gradients, first_generator_model.trainable_variables))
    #first_generator_optimizer_bias.apply_gradients(zip(l1_l2_gradients[-1:], first_generator_model.trainable_variables[-1:]))
    return (gen_cost_l1, gen_cost_l2, gen_cost_l_inf, gradient_gen_l1, gradient_gen_l2, gradient_gen_l_inf, sr_tiles)