# Parameters which have to be set before each run 

# Choose the mode of the run (only one should not be commented)
#mode = 'testing_network'
mode = 'training'

# Set directory for the Output
directory = '/u/ivkos/sr/'
run_dir = directory+'primitive_gan_avg1_scaled2'

# Set stack and batch size
stack_size = 5120
BATCH_SIZE = 64

# Set the size of the input tiles
x_dim = 64
y_dim = 64
z_dim = 64

# Set the size for the label tiles
x_dim_out = 64
y_dim_out = 64
z_dim_out = 64

# Location training tiles make sure that the training tiles are numbered and mark the number by {} for correct formating
loc_training_tiles = directory+'TrainingTiles/lr_no_scale/lr_cube{}.npy'

# Location label tiles keep numbering consitend with the traing tiles
loc_label_tiles = directory+'TrainingTiles/hr_no_scale/hr_cube{}.npy'

# Import the desired model
from models import simple_gan as model

# State whether the given model is a multipass or single pass model
mult_pass = False

# State whether you want to continue training an already pretrained network
continue_training = True

# If the above is set to 'True' state in which folder the pretrained network is located
pretrained_dir = directory+'primitive_gan_avg1_scaled/'

# Import the cost functions (all the cost functions should be in the same file and this should not be changed)
from loss import loss_gan_only as loss

# Set the loss functions you want to use in this training run
if mult_pass:
    first_gen_loss = loss.first_generator_loss
    first_crit_loss = loss.first_critic_loss

    second_gen_loss = loss.second_generator_loss
    second_crit_loss = loss.second_critic_loss
else:
    first_gen_loss = loss.first_generator_loss
    first_crit_loss = loss.first_critic_loss

# State whether a scheduled step size decay should be used in the optimizer
step_size_decay = False

# Define the weight decays for that the import of tensorflow is needed
import tensorflow as tf 
first_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=6000, decay_rate=0.96)
first_schedule_gen_bias = tf.keras.optimizers.schedules.ExponentialDecay(0.00005, decay_steps=6000, decay_rate=0.96)
first_schedule_crit = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=6000, decay_rate=0.96)

second_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=6000, decay_rate=0.96)
second_schedule_gen_bias = tf.keras.optimizers.schedules.ExponentialDecay(0.00005, decay_steps=6000, decay_rate=0.96)
second_schedule_crit = tf.keras.optimizers.schedules.ExponentialDecay(0.0005, decay_steps=6000, decay_rate=0.96)

# Define the optimizers for the networks (usually not changed)
# Define the optimizers for the generator and critic network for the first network or the only network for single pass
first_generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001, beta_1 = 0.5, beta_2=0.999)
#first_generator_bias_optimizer = tf.keras.optimizers.Adam(learning_rate =  0.0001, beta_1 = 0.5, beta_2=0.999)
first_critic_optimizer = tf.keras.optimizers.Adam(learning_rate =  0.00001, beta_1 = 0.5, beta_2=0.999)

# Define the optimizers for the generator and critic network for the second network
second_generator_optimizer = tf.keras.optimizers.Adam(learning_rate =  0.0001, beta_1 = 0.5, beta_2=0.099)
second_generator_bias_optimizer = tf.keras.optimizers.Adam(learning_rate =  0.0001, beta_1 = 0.5, beta_2=0.099)
second_critic_optimizer = tf.keras.optimizers.Adam(learning_rate =  0.0001, beta_1 = 0.5, beta_2=0.999)

# Number of critic updates for each generator update
k_crit = 10

