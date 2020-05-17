# Short script which returns the description of a given model and plots its layer structure

import tensorflow as tf 
import parameters as p
from models import model_3d
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers


first_generator_input = layers.Input(shape=(None,None,None,1))
first_generator_model = tf.keras.Model(inputs = first_generator_input, outputs = model_3d.first_generator(first_generator_input))

# first_critic_input = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
# first_critic_model = tf.keras.Model(inputs = first_critic_input, outputs = p.model.first_critic(first_critic_input, first_critic_input))

first_generator_model.summary()
# first_critic_model.summary()
plot_model(first_generator_model, to_file='first_gen_residual_learning.png')
# plot_model(first_critic_model, to_file='first_crit_residual_learning.png')


exit()




if p.mult_pass:
    # Define the generator and critic model for the first gen, crit
    first_generator_input = layers.Input(shape=(None,None,1))
    first_generator_model = tf.keras.Model(inputs = first_generator_input, outputs = p.model.first_generator(first_generator_input))

    first_critic_input = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    first_critic_model = tf.keras.Model(inputs = first_critic_input, outputs = p.model.first_critic(first_critic_input, first_critic_input))

    # Define the generator and the critic model for the second gen, crit
    second_generator_input = layers.Input(shape=(None,None,1))
    second_generator_model = tf.keras.Model(inputs = second_generator_input, outputs = p.model.second_generator(second_generator_input))

    second_critic_input = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    second_critic_model = tf.keras.Model(inputs = second_critic_input, outputs = p.model.second_critic(second_critic_input, second_critic_input))
    first_generator_model.summary()
    first_critic_model.summary()
    plot_model(first_generator_model, to_file='first_gen_residual_learning.png')
    plot_model(first_critic_model, to_file='first_crit_residual_learning.png')

    second_generator_model.summary()
    second_critic_model.summary()
    plot_model(second_generator_model, to_file='second_gen_residual_learning.png')
    plot_model(second_critic_model, to_file='second_crit_residual_learning.png')
else:
    # Define the generator and critic model
    first_generator_input = layers.Input(shape=(None,None,1))
    first_generator_model = tf.keras.Model(inputs = first_generator_input, outputs = p.model.generator(first_generator_input))

    first_critic_input = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    first_critic_model = tf.keras.Model(inputs = first_critic_input, outputs = p.model.critic(first_critic_input))

