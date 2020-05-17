# Define the training steps for thetraining procedure 

import tensorflow as tf 

tf.random.set_seed(42)


# Define the training step using the tf function to compile it (first network)
@tf.function
def first_train_step(input_tiles, labels, first_generator_model, first_critic_model, 
                    first_generator_loss, first_critic_loss, first_generator_optimizer, first_critic_optimizer):
    # with tf.GradientTape() as gen_tape, tf.GradientTape() as crit_tape, tf.GradientTape() as sr_tape, tf.GradientTape() as hr_tape, tf.GradientTape() as grad_tape, \
    #     tf.GradientTape() as gen_gan_tape, tf.GradientTape() as gen_inf_tape:
    with tf.GradientTape() as gen_tape, tf.GradientTape() as crit_tape:

        sr_tiles = first_generator_model(input_tiles)

        #gan_loss, l_loss, gen_loss = first_generator_loss(labels, sr_tiles, input_tiles, first_critic_model)
        gen_gan_loss, gen_inf_loss, gen_loss = first_generator_loss(labels, sr_tiles, input_tiles, first_critic_model)

        sr_loss, hr_loss, grad_loss, crit_loss = first_critic_loss(labels, sr_tiles, input_tiles, first_critic_model)
        #crit_loss = first_critic_loss(labels, sr_tiles, input_tiles, first_critic_model)

    gradients_of_generator = gen_tape.gradient(gen_loss, first_generator_model.trainable_variables)
    #gradients_gan_of_generator = gen_gan_tape.gradient(gen_gan_loss, first_generator_model.trainable_variables)
    #gradients_inf_of_generator = gen_inf_tape.gradient(gen_inf_loss, first_generator_model.trainable_variables)


    gradients_of_critic = crit_tape.gradient(crit_loss, first_critic_model.trainable_variables)
    #gradients_of_sr = sr_tape.gradient(sr_loss, first_critic_model.trainable_variables)
    #gradients_of_hr = hr_tape.gradient(hr_loss, first_critic_model.trainable_variables)
    #gradients_of_grad = grad_tape.gradient(grad_loss, first_critic_model.trainable_variables)
   
    first_generator_optimizer.apply_gradients(zip(gradients_of_generator, first_generator_model.trainable_variables))
    first_critic_optimizer.apply_gradients(zip(gradients_of_critic, first_critic_model.trainable_variables))

    #return (gen_cost, crit_cost, gradient_gen, gradient_crit, sr_tiles, gan_loss, l_loss, sr_loss, hr_loss, grad_loss, gradients_of_generator, gradients_of_critic, 
    #gradients_lx_generator, gradients_gan_generator)
    return (gen_loss, crit_loss, gradients_of_generator, gradients_of_critic)
    #return (gen_cost, crit_cost, gradient_gen, gradient_crit, sr_tiles)



# Define the training step for only the critic using the tf function to compile it (first network)
@tf.function
def first_train_step_crit(input_tiles, labels, first_generator_model, first_critic_model, first_critic_loss, first_critic_optimizer):

    #with tf.GradientTape() as crit_tape, tf.GradientTape() as sr_tape, tf.GradientTape() as hr_tape, tf.GradientTape() as grad_tape:
    with tf.GradientTape() as crit_tape:
        sr_tiles = first_generator_model(input_tiles)
        sr_loss, hr_loss, grad_loss, crit_loss = first_critic_loss(labels, sr_tiles, input_tiles, first_critic_model)
        #crit_loss = first_critic_loss(labels, sr_tiles, input_tiles, first_critic_model)

    gradients_of_critic = crit_tape.gradient(crit_loss, first_critic_model.trainable_variables)
    #gradients_of_sr = sr_tape.gradient(sr_loss, first_critic_model.trainable_variables)
    #gradients_of_hr = hr_tape.gradient(hr_loss, first_critic_model.trainable_variables)
    #gradients_of_grad = grad_tape.gradient(grad_loss, first_critic_model.trainable_variables)

    first_critic_optimizer.apply_gradients(zip(gradients_of_critic, first_critic_model.trainable_variables))
    
    return (crit_loss, gradients_of_critic)
    #return (crit_cost, gradient_crit, sr_tiles, gradients_of_critic, sr_loss, hr_loss, grad_loss)



# Define the training step using the tf function to compile it (second network)
@tf.function
def second_train_step(input_tiles, labels, gen_cost, crit_cost, gradient_crit, gradient_gen, second_generator_model, second_critic_model, second_generator_loss, second_critic_loss, second_generator_optimizer, second_critic_optimizer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        sr_tiles = second_generator_model(input_tiles)
        gen_loss = second_generator_loss(labels, sr_tiles, input_tiles, second_critic_model)
        disc_loss = second_critic_loss(labels, sr_tiles, input_tiles, second_critic_model)
        gen_cost = tf.add(gen_loss,gen_cost)
        crit_cost = tf.add(disc_loss, crit_cost)

    gradients_of_generator = gen_tape.gradient(gen_loss, second_generator_model.trainable_variables)
    gradients_of_critic = disc_tape.gradient(disc_loss, second_critic_model.trainable_variables)

    #Save the total value of the gradients
    for variable in gradients_of_generator:
        tot_grad_gen = tf.reduce_sum(tf.abs(variable))
        gradient_gen = tf.add(tot_grad_gen, gradient_gen) 
    for variable in gradients_of_critic:
        tot_grad_crit = tf.reduce_sum(tf.abs(variable))
        gradient_crit = tf.add(tot_grad_crit, gradient_crit)
   

    second_generator_optimizer.apply_gradients(zip(gradients_of_generator, second_generator_model.trainable_variables))
    second_critic_optimizer.apply_gradients(zip(gradients_of_critic, second_critic_model.trainable_variables))
    return (gen_cost, crit_cost, gradient_gen, gradient_crit)

# Define the training step for only the critic using the tf function to compile it (second network)
@tf.function
def second_train_step_crit(input_tiles, labels, crit_cost, gradient_crit, second_generator_model, second_critic_model, second_critic_loss, second_critic_optimizer):
    with tf.GradientTape() as disc_tape:
        sr_tiles = second_generator_model(input_tiles)
        disc_loss = second_critic_loss(labels, sr_tiles, input_tiles, second_critic_model)
        crit_cost = tf.add(disc_loss, crit_cost)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, second_critic_model.trainable_variables)

    #Save the total value of the gradients
    for variable in gradients_of_discriminator:
        tot_grad_crit = tf.reduce_sum(tf.abs(variable))
        gradient_crit = tf.add(tot_grad_crit, gradient_crit)
   
    second_critic_optimizer.apply_gradients(zip(gradients_of_discriminator, second_critic_model.trainable_variables))
    return (crit_cost, gradient_crit)

