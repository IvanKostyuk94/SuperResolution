#!/usr/bin/python3

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf 
import numpy as np 
from matplotlib import pyplot as plt
from tensorflow.keras import layers

# Needs to be set to avoid conflicts when adding tensors
tf.keras.backend.set_floatx('float32')

#Things that have gan_k5to be adjusted every!!!! time
training_run_name = 'gan_scale_no_res' # Very important should be changed every time if its not a second training session simply choose model_name
try:
    os.mkdir('/u/ivkos/sr/Output/{}'.format(training_run_name))
except:
    pass
model_name = 'convolutions5x5' # Choose the model file which contains the network models trained in the given run
from models.convolutions5x5 import * #Make sure to insert model_name here as well!



#Parameters to pay attention to before each run:
Input = '/u/ivkos/sr/GriddedSimulationsTesting/test256_grid512.npy' # This should not change in most runs
output_name = training_run_name+'_sr_output' # Should be changed if you don't want to have the default name
factor = 2 # Will probably become redundant as the sr factors will be determined by the models directly

model_path = '/u/ivkos/sr/code/models/{}.py'.format(model_name)
#os.system(f'cp {model_path} .')

simulation_cube_blurred = np.load(Input)

# Scaling and unscaling functions for the data
def scale(data, epsilon=1e-9):
    return np.log(data+epsilon)/25+1
def unscale(data, epsilon=1e-9):
    return (np.exp(25*(data-1))-epsilon)

def scale_2(data, epsilon=1e-9):
    return np.log(data+epsilon)/25
def unscale_2(data, epsilon=1e-9):
    return (np.exp(25*(data))-epsilon)

def scale_3(data, epsilon=1e-9):
    return np.log(data+epsilon)
def unscale_3(data, epsilon=1e-9):
    return (np.exp(data)-epsilon)

def scale_4(data, epsilon=1e-9):
    return np.log(data+epsilon)+25
def unscale_4(data, epsilon=1e-9):
    return (np.exp(data-25)-epsilon)

simulation_cube_blurred = scale_2(simulation_cube_blurred)
simulation_cube_blurred = simulation_cube_blurred.reshape(simulation_cube_blurred.shape[0],simulation_cube_blurred.shape[1],simulation_cube_blurred.shape[2],1)

generator_input = layers.Input(shape=(None,None,1))

first_generator_model = tf.keras.Model(inputs = generator_input, outputs = first_generator(generator_input))
second_generator_model = tf.keras.Model(inputs = generator_input, outputs = second_generator(generator_input))

# Load the latest checkpoints
ckpt_dir = '/u/ivkos/sr/{}/checkpoints/'.format(training_run_name)
first_ckpt_dir = ckpt_dir + 'first_network_checkpoints/'
second_ckpt_dir = ckpt_dir + 'second_network_checkpoints/'

latest_first = tf.train.latest_checkpoint(first_ckpt_dir)
#latest_second = tf.train.latest_checkpoint(second_ckpt_dir)
print(latest_first)
#print(latest_second)

first_checkpoint = tf.train.Checkpoint(generator=first_generator_model)
first_checkpoint.restore(latest_first)
#second_checkpoint = tf.train.Checkpoint(generator=second_generator_model)
#second_checkpoint.restore(latest_second)

multiple_checkpoints = False

if multiple_checkpoints:
    for i in range(37):
        first_checkpoint = tf.train.Checkpoint(generator=first_generator_model)
        first_checkpoint.restore(first_ckpt_dir+'ckpt-{}'.format(i+1))
        #second_checkpoint = tf.train.Checkpoint(generator=second_generator_model)
        #second_checkpoint.restore(second_ckpt_dir+'ckpt-{}'.format(i*10+1))

        print('The max of the simulation cube is: {}'.format(np.amax(simulation_cube_blurred)))
        print('The min of the simulation cube is: {}'.format(np.amin(simulation_cube_blurred)))
        print('The average of the simulation cube is: {}'.format(np.average(simulation_cube_blurred)))
        first_pass = first_generator_model.predict(simulation_cube_blurred)
        print('The max of the simulation cube after the first pass is: {}'.format(np.amax(first_pass)))
        print('The min of the simulation cube after the first pass is: {}'.format(np.amin(first_pass)))
        print('The average of the simulation cube after the first pass is: {}'.format(np.average(first_pass)))
        second_input = tf.transpose(first_pass, [1,0,2,3])
        second_pass = second_generator_model.predict(second_input)
        second_pass = tf.transpose(second_pass, [1,0,2,3])
        print('The max of the simulation cube after the second pass is: {}'.format(np.amax(second_pass)))
        print('The min of the simulation cube after the second pass is: {}'.format(np.amin(second_pass)))
        print('The average of the simulation cube after the second pass is: {}'.format(np.average(second_pass)))

        #np.save('/u/ivkos/sr/Output/{}/{}_ckpt{}'.format(training_run_name, output_name, i+1), unscale_2(second_pass[:,:,:,0]))
        np.save('/u/ivkos/sr/Output/{}/{}_ckpt{}'.format(training_run_name, output_name, i+1), second_pass[:,:,:,0])
        #np.save('/u/ivkos/sr/Output/{}/{}_ckpt{}'.format(training_run_name, output_name, i*10+1), unscale_2(first_pass[:,:,:,0]))

else:
    #first_checkpoint = tf.train.Checkpoint(generator=first_generator_model)
    #first_checkpoint.restore(latest_first)
    #second_checkpoint = tf.train.Checkpoint(generator=second_generator_model)
    #second_checkpoint.restore(latest_second)
    print('The max of the simulation cube is: {}'.format(np.amax(simulation_cube_blurred)))
    print('The min of the simulation cube is: {}'.format(np.amin(simulation_cube_blurred)))
    print('The average of the simulation cube is: {}'.format(np.average(simulation_cube_blurred)))
    first_pass = first_generator_model.predict(simulation_cube_blurred)
    print('The max of the simulation cube after the first pass is: {}'.format(np.amax(first_pass)))
    print('The min of the simulation cube after the first pass is: {}'.format(np.amin(first_pass)))
    print('The average of the simulation cube after the first pass is: {}'.format(np.average(first_pass)))
    # second_input = tf.transpose(first_pass, [1,0,2,3])
    # second_pass = second_generator_model.predict(second_input)
    # second_pass = tf.transpose(second_pass, [1,0,2,3])
    # print('The max of the simulation cube after the second pass is: {}'.format(np.amax(second_pass)))
    # print('The min of the simulation cube after the second pass is: {}'.format(np.amin(second_pass)))
    # print('The average of the simulation cube after the second pass is: {}'.format(np.average(second_pass)))
    np.save('/u/ivkos/sr/Output/{}/{}'.format(training_run_name, output_name), unscale_2(first_pass[:,:,:,0]))
    #np.save('/u/ivkos/sr/Output/{}/{}'.format(training_run_name, output_name), unscale_2(first_pass[:,:,:,0]))