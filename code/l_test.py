import tensorflow as tf 
import numpy as np 
import time
import datetime
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers

import parameters as p

from steps import steps_l_test as steps

from loss import l1_l2 as lossfunc

#from models.convolutions5x5 import *
from models.test_model import *
from models.convolutions5x5 import *

directory = '/u/ivkos/sr/code/results_test/'
run_dir = directory+'l_test3'

loss_dir = run_dir+'/loss/'
grad_dir = run_dir+'/grad/'
check_dir = run_dir + '/checkpoints/'

os.mkdir(run_dir)
os.mkdir(loss_dir)
os.mkdir(grad_dir)
os.mkdir(check_dir)

tf.random.set_seed(13)
tf.keras.backend.set_floatx('float32')

stack_size = 5120
BATCH_SIZE = 64

train_stack = np.zeros((stack_size*p.x_dim, p.y_dim, p.z_dim,1)).astype(np.float32)
label_stack = np.zeros((stack_size*p.x_dim_out, p.y_dim_out, p.z_dim_out,1)).astype(np.float32)

main_dir = '/u/ivkos/sr/'

# Location training tiles make sure that the training tiles are numbered and mark the number by {} for correct formating
loc_training_tiles = main_dir+'TrainingTiles/lr_no_scale/lr_cube{}.npy'

# Location label tiles keep numbering consitend with the traing tiles
loc_label_tiles = main_dir+'TrainingTiles/hr_no_scale/hr_cube{}.npy'

def scale(data, epsilon=1e-9):
    return np.log(data+epsilon)/25
def unscale(data, epsilon=1e-9):
    return (np.exp(25*(data))-epsilon)

def scale_s(data, a=1e-5):
    return 2*data/(data+a)-1
def unscale_s(data, a=1e-5):
    return a*(data+1)/(data+3.)

for i in range(stack_size):
    train_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,0] = scale(np.load(loc_training_tiles.format(i)).astype(np.float32))
    label_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,0] = scale(np.load(loc_label_tiles.format(i)).astype(np.float32))

path_to_data = '/u/ivkos/sr/GriddedSimulationsTraining/'
sim256_grid512_path = path_to_data+'256_grid512.npy'

sim = np.load(sim256_grid512_path)
test_slice = scale(sim[42,...])
test_slice = test_slice.reshape(1, test_slice.shape[0], test_slice.shape[1],1)

def conv_uint(data, test_slice=test_slice):
    max_val = np.max(test_slice)
    min_val = np.min(test_slice)
    alpha = (data-min_val)/(max_val-min_val)
    return (alpha*254).astype(np.uint8)

# Batch the data
train_dataset_first = tf.data.Dataset.from_tensor_slices(train_stack).batch(BATCH_SIZE)
label_dataset_first = tf.data.Dataset.from_tensor_slices(label_stack).batch(BATCH_SIZE)
del train_stack
del label_stack

#train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = run_dir+'/logs/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

generator_input = layers.Input(shape=(None,None,1))
generator_model = tf.keras.Model(inputs = generator_input, outputs = first_generator(generator_input))

checkpoint_prefix = os.path.join(check_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator_model)

tot_epochs = 5

epoch = 0
tot_counter = 0
img_counter=0
for epoch in range(tot_epochs):
    batch_counter = 0
    for train_batch, label_batch in zip(train_dataset_first, label_dataset_first):
        tot_counter = 5120*epoch+batch_counter
        print('Epoch {}, batch {}'.format(epoch,batch_counter))
        loss, gradients = steps.train_step(train_batch, label_batch, generator_model, lossfunc.generator_loss, p.first_generator_optimizer)
        if tot_counter % 50 == 0:
            img = generator_model.predict(test_slice)
            img_counter += 1

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', loss, step=tot_counter)
            tensor_counter=0
            for tensor in generator_model.trainable_variables:
                tf.summary.histogram('tensor {}'.format(tensor_counter), tensor, step=tot_counter)
                tensor_counter += 1

            tf.summary.image('sr slice', conv_uint(img), step=img_counter)
            tf.summary.image('lr slice', conv_uint(test_slice), step=0)

            train_summary_writer.flush()
        batch_counter += 1
    epoch += 1