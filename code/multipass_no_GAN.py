import os 
import parameters as p
from steps import steps_no_GAN as step

# Set the device to CPU for testing on the Laptop as the GPU has to little memory 
if p.mode == 'testing_results':
    os.environ["CUDA_VISIBLE_DEVICES"]="-1" 

import tensorflow as tf
import numpy as np 
import shutil
from tensorflow.keras import layers
import time
from tensorflow import keras
from sys import getsizeof

# Needs to be set to avoid conflicts when adding tensors
tf.keras.backend.set_floatx('float32')

# Generate all the subfolders for a given run
run_dir = p.run_dir
try:
    os.mkdir(run_dir)
except:
    # In case the folder already exist create another one by adding a 2 to its name
    run_dir = run_dir+str(int(time.time()))
    os.mkdir(run_dir)

loss_dir = run_dir+'/loss/'
grad_dir = run_dir+'/grad/'
check_dir = run_dir + '/checkpoints/'

first_checkpoint_dir = check_dir+'first_network_checkpoints/'

if p.mult_pass:
    second_checkpoint_dir = check_dir+'second_network_checkpoints/'

# Create all the folders defined above


os.mkdir(loss_dir)
os.mkdir(grad_dir)
os.mkdir(check_dir)

os.mkdir(first_checkpoint_dir)
os.mkdir(second_checkpoint_dir)

stack_size = p.stack_size
BATCH_SIZE = p.BATCH_SIZE

train_stack = np.zeros((stack_size*p.x_dim, p.y_dim, p.z_dim,1)).astype(np.float32)
label_stack = np.zeros((stack_size*p.x_dim_out, p.y_dim_out, p.z_dim_out,1)).astype(np.float32)
for i in range(stack_size):
    train_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,0] = np.load(p.loc_training_tiles.format(i)).astype(np.float32)
    label_stack[i*p.x_dim_out:(i+1)*p.x_dim_out,:,:,0] = np.load(p.loc_label_tiles.format(i)).astype(np.float32)

# Batch the data
train_dataset_first = tf.data.Dataset.from_tensor_slices(train_stack).batch(BATCH_SIZE)
label_dataset_first = tf.data.Dataset.from_tensor_slices(label_stack).batch(BATCH_SIZE)

del train_stack
del label_stack

if p.mult_pass:
    # Define the generator and critic model for the first gen, crit
    first_generator_input = layers.Input(shape=(None,None,1))
    first_generator_model = tf.keras.Model(inputs = first_generator_input, outputs = p.model.first_generator(first_generator_input))

    first_critic_input1 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    first_critic_input2 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    first_critic_model = tf.keras.Model(inputs = (first_critic_input1, first_critic_input2), outputs = p.model.first_critic(first_critic_input1, first_critic_input2))

    # Define the generator and the critic model for the second gen, crit
    second_generator_input = layers.Input(shape=(None,None,1))
    second_generator_model = tf.keras.Model(inputs = second_generator_input, outputs = p.model.second_generator(second_generator_input))

    second_critic_input1 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    second_critic_input2 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    second_critic_model = tf.keras.Model(inputs = (second_critic_input1, second_critic_input2), outputs = p.model.second_critic(second_critic_input1, second_critic_input2))
else:
    # Define the generator and critic model
    first_generator_input = layers.Input(shape=(None,None,1))
    first_generator_model = tf.keras.Model(inputs = first_generator_input, outputs = p.model.generator(first_generator_input))

    first_critic_input1 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    first_critic_input2 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
    first_critic_model = tf.keras.Model(inputs = (first_critic_input1, first_critic_input2), outputs = p.model.first_critic(first_critic_input1, first_critic_input2))

if p.continue_training:
    if p.mult_pass:
        # Load the pretrained networks to continue training
        ckpt_dir = p.pretrained_dir+'checkpoints/'
        first_ckpt_dir = ckpt_dir + 'first_network_checkpoints/'
        second_ckpt_dir = ckpt_dir + 'second_network_checkpoints/'

        latest_first = tf.train.latest_checkpoint(first_ckpt_dir)
        latest_second = tf.train.latest_checkpoint(second_ckpt_dir)
        first_checkpoint = tf.train.Checkpoint(generator=first_generator_model, critic = first_critic_model)
        first_checkpoint.restore(latest_first)
        second_checkpoint = tf.train.Checkpoint(generator=second_generator_model, critic = second_critic_model)
        second_checkpoint.restore(latest_second)
    else:
        ckpt_dir = p.pretrained_dir+'checkpoints/'
        first_ckpt_dir = ckpt_dir + 'first_network_checkpoints/'

        latest = tf.train.latest_checkpoint(first_ckpt_dir)
        checkpoint = tf.train.Checkpoint(generator=first_generator_model, critic = first_critic_model)
        checkpoint.restore(latest)

if p.mult_pass:
    first_checkpoint_prefix = os.path.join(first_checkpoint_dir, "ckpt")
    second_checkpoint_prefix = os.path.join(second_checkpoint_dir, "ckpt")
    first_checkpoint = tf.train.Checkpoint(generator=first_generator_model, critic=first_critic_model)
    second_checkpoint = tf.train.Checkpoint(generator=second_generator_model, critic=second_critic_model)
else: 
    first_checkpoint_prefix = os.path.join(first_checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator=first_generator_model, critic=first_critic_model)

# Define the training loop
def train(input_set, label_set, epochs):
    first_gen_cost_l1_list=[]
    first_gen_cost_l2_list=[]
    first_gen_cost_l_inf_list=[]
    first_grad_gen_l1_list=[]
    first_grad_gen_l2_list=[]
    first_grad_gen_l_inf_list=[]
    
    second_gen_cost_l1_list=[]
    second_gen_cost_l2_list=[]
    second_gen_cost_l_inf_list=[]
    second_grad_gen_l1_list=[]
    second_grad_gen_l2_list=[]
    second_grad_gen_l_inf_list=[]

    for epoch in range(epochs):
        start = time.time()

        first_gen_cost_l1 = 0.
        first_gen_cost_l2 = 0.
        first_gen_cost_l_inf = 0.
        first_grad_gen_l1 = 0.
        first_grad_gen_l2 = 0.
        first_grad_gen_l_inf = 0.

        second_gen_cost_l1 = 0.
        second_gen_cost_l2 = 0.
        second_gen_cost_l_inf = 0.
        second_grad_gen_l1 = 0.
        second_grad_gen_l2 = 0.
        second_grad_gen_l_inf = 0.
        
        batch_counter = 0
        for train_batch, label_batch in zip(train_dataset_first, label_dataset_first):
            batch_counter += 1
            (first_gen_cost_l1, first_gen_cost_l2, first_gen_cost_l_inf,
            first_grad_gen_l1, first_grad_gen_l2, first_grad_gen_l_inf,
            sr_tiles) = step.first_train_step(train_batch, label_batch, 
                                            first_gen_cost_l1, first_gen_cost_l2, first_gen_cost_l_inf,
                                            first_grad_gen_l1, first_grad_gen_l2, first_grad_gen_l_inf,
                                            first_generator_model, p.first_gen_loss,  
                                            p.first_generator_optimizer, p.first_generator_bias_optimizer)

            # Turn the training tile around
            second_batch = tf.transpose(sr_tiles, [1,0,2,3])
            second_labels = tf.transpose(label_batch, [1,0,2,3])

            (second_gen_cost_l1, second_gen_cost_l2, second_gen_cost_l_inf,
            second_grad_gen_l1, second_grad_gen_l2, second_grad_gen_l_inf) = step.second_train_step(second_batch, second_labels, 
                                                                            second_gen_cost_l1, second_gen_cost_l2, second_gen_cost_l_inf,
                                                                            second_grad_gen_l1, second_grad_gen_l2, second_grad_gen_l_inf,
                                                                            second_generator_model, p.second_gen_loss, 
                                                                            p.second_generator_optimizer, p.second_generator_bias_optimizer)
            
        #Save cost and gradient for diagnostics
        print(batch_counter)
        first_gen_cost_l1_list.append(first_gen_cost_l1/batch_counter)
        first_gen_cost_l2_list.append(first_gen_cost_l2/batch_counter)
        first_gen_cost_l_inf_list.append(first_gen_cost_l_inf/batch_counter)
        first_grad_gen_l1_list.append(first_grad_gen_l1/batch_counter)
        first_grad_gen_l2_list.append(first_grad_gen_l2/batch_counter)
        first_grad_gen_l_inf_list.append(first_grad_gen_l_inf/batch_counter)
        
        second_gen_cost_l1_list.append(second_gen_cost_l1/batch_counter)
        second_gen_cost_l2_list.append(second_gen_cost_l2/batch_counter)
        second_gen_cost_l_inf_list.append(second_gen_cost_l_inf/batch_counter)
        second_grad_gen_l1_list.append(second_grad_gen_l1/batch_counter)
        second_grad_gen_l2_list.append(second_grad_gen_l2/batch_counter)
        second_grad_gen_l_inf_list.append(second_grad_gen_l_inf/batch_counter)
        
        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            first_checkpoint.save(file_prefix = first_checkpoint_prefix)
            second_checkpoint.save(file_prefix = second_checkpoint_prefix)
            np.savetxt(loss_dir+'first_gen_cost_l1.txt', np.array(first_gen_cost_l1_list))
            np.savetxt(loss_dir+'first_gen_cost_l2.txt', np.array(first_gen_cost_l2_list))
            np.savetxt(loss_dir+'first_gen_cost_l_inf.txt', np.array(first_gen_cost_l_inf_list))
            np.savetxt(grad_dir+'first_gen_grad_l1.txt', np.array(first_grad_gen_l1_list))
            np.savetxt(grad_dir+'first_gen_grad_l2.txt', np.array(first_grad_gen_l2_list))
            np.savetxt(grad_dir+'first_gen_grad_l_inf.txt', np.array(first_grad_gen_l_inf_list))

            np.savetxt(loss_dir+'second_gen_cost_l1.txt', np.array(second_gen_cost_l1_list))
            np.savetxt(loss_dir+'second_gen_cost_l2.txt', np.array(second_gen_cost_l2_list))
            np.savetxt(loss_dir+'second_gen_cost_l_inf.txt', np.array(second_gen_cost_l_inf_list))
            np.savetxt(grad_dir+'second_gen_grad_l1.txt', np.array(second_grad_gen_l1_list))
            np.savetxt(grad_dir+'second_gen_grad_l2.txt', np.array(second_grad_gen_l2_list))
            np.savetxt(grad_dir+'second_gen_grad_l_inf.txt', np.array(second_grad_gen_l_inf_list))

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

if p.mode != 'testing_results':
    if p.mode == 'training':
        train(train_dataset_first, label_dataset_first, 1000)    
    if p.mode == 'testing_network':
        train(train_dataset_first, label_dataset_first, 30)
