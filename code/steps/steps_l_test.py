# Define the training steps for thetraining procedure 

import tensorflow as tf 

# Define the training step using the tf function to compile it (first network)
@tf.function
def train_step(input_tiles, labels, 
                    generator_model, generator_loss, 
                    generator_optimizer):
    
    with tf.GradientTape() as gen_tape:
        sr_tiles = generator_model(input_tiles)
        loss = generator_loss(labels, sr_tiles)

    gradients = gen_tape.gradient(loss, generator_model.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))
    return (loss, gradients)