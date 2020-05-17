import os 
import parameters as p
from steps import steps_l_test as step

import tensorflow as tf
import numpy as np 
import shutil
import datetime

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

#loss_dir = run_dir+'/loss/'
#grad_dir = run_dir+'/grad/'
check_dir = run_dir + '/checkpoints/'

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = run_dir+'/logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

# Create all the folders defined above
#os.mkdir(loss_dir)
#os.mkdir(grad_dir)
os.mkdir(check_dir)

stack_size = p.stack_size
BATCH_SIZE = p.BATCH_SIZE

train_stack = np.zeros((stack_size*p.x_dim, p.y_dim, p.z_dim,2)).astype(np.float32)
label_stack = np.zeros((stack_size*p.x_dim_out, p.y_dim_out, p.z_dim_out,2)).astype(np.float32)

def scale(data, epsilon=1e-9):
    return np.log(data+epsilon)/25
def unscale(data, epsilon=1e-9):
    return (np.exp(25*(data))-epsilon)

def scale_s(data, a=1e-5):
    return 2*data/(data+a)-1
def unscale_s(data, a=1e-5):
    return a*(data+1.)/(1.-data)

for i in range(stack_size):
    train_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,0] = scale(np.load(p.loc_training_tiles.format(i)).astype(np.float32))
    label_stack[i*p.x_dim_out:(i+1)*p.x_dim_out,:,:,0] = scale(np.load(p.loc_label_tiles.format(i)).astype(np.float32))
    
    train_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,1] = np.load(p.loc_training_tiles.format(i)).astype(np.float32)
    label_stack[i*p.x_dim_out:(i+1)*p.x_dim_out,:,:,1] = np.load(p.loc_label_tiles.format(i)).astype(np.float32)

# Batch the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_stack).batch(BATCH_SIZE)
label_dataset = tf.data.Dataset.from_tensor_slices(label_stack).batch(BATCH_SIZE)

del train_stack
del label_stack

# Define the generator and critic model for the first gen, crit
generator_input = layers.Input(shape=(None,None,2))
generator_model = tf.keras.Model(inputs = generator_input, outputs = p.model.generator(generator_input))

if p.continue_training:
    # Load the pretrained networks to continue training
    ckpt_dir = p.pretrained_dir+'checkpoints/'

    latest = tf.train.latest_checkpoint(ckpt_dir)
    checkpoint = tf.train.Checkpoint(generator=generator_model)
    checkpoint.restore(latest)

checkpoint_prefix = os.path.join(check_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator_model)

path_to_data = '/u/ivkos/sr/GriddedSimulationsTraining/'
sim256_grid512_path = path_to_data+'256_grid512.npy'

sim = np.load(sim256_grid512_path)
test_slice = scale_s(sim[42,...])
test_slice = test_slice.reshape(1, test_slice.shape[0], test_slice.shape[1],1)

def conv_uint(data, test_slice=test_slice):
    max_val = np.max(test_slice)
    min_val = np.min(test_slice)
    alpha = (data-min_val)/(max_val-min_val)
    return (alpha*254).astype(np.uint8)

# Define the training loop
def train(input_set, label_set, epochs):
    tot_counter = 0
    for epoch in range(epochs):
        start = time.time()
        
        batch_counter = 0
        for train_batch, label_batch in zip(train_dataset, label_dataset):
            tot_counter += 1

            (gen_loss, gen_gradients) = step.train_step(train_batch, label_batch,
                                                        generator_model, p.generator_loss,
                                                        p.first_generator_optimizer)

            if batch_counter == 0:
                img = generator_model.predict(test_slice)

            with train_summary_writer.as_default():
                tf.summary.scalar('generator loss', gen_loss, step=tot_counter)

                gen_tensor_counter=0
                for tensor in generator_model.trainable_variables:
                    tf.summary.histogram('generator tensor {}'.format(gen_tensor_counter), tensor, step=epoch)
                    gen_tensor_counter += 1

                tf.summary.image('sr slice', conv_uint(img), step=epoch)
                tf.summary.image('lr slice', conv_uint(test_slice), step=0)

            batch_counter += 1

        # Save the model every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

train(train_dataset, label_dataset, 10000)  