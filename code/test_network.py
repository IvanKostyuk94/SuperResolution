import tensorflow as tf 
import numpy as np 
import time
import datetime
import os
from matplotlib import pyplot as plt
from tensorflow.keras import layers

import parameters as p


from steps import steps_no_GAN
from steps import training_steps
from steps import steps_3d

from loss import loss_gan_only
from loss import loss_gan_lx
from loss import l1_l2
from loss import lx_loss_3d

#from models.convolutions5x5 import *
from models.test_model import *

directory = '/u/ivkos/sr/code/results_test/'
run_dir = directory+'long_test1'

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

for i in range(stack_size):
    train_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,0] = np.load(p.loc_training_tiles.format(i)).astype(np.float32)
    label_stack[i*p.x_dim:(i+1)*p.x_dim,:,:,0] = np.load(p.loc_label_tiles.format(i)).astype(np.float32)

def scale(data, epsilon=1e-9):
    return np.log(data+epsilon)/25
def unscale(data, epsilon=1e-9):
    return (np.exp(25*(data))-epsilon)

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
train_log_dir = run_dir+'/logs/gradient_tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)



generator_input = layers.Input(shape=(None,None,1))
generator_model = tf.keras.Model(inputs = generator_input, outputs = generator(generator_input))

sr_input = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
hr_input = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
critic_model = tf.keras.Model(inputs = (sr_input, hr_input), outputs = critic(sr_input, hr_input))

checkpoint_prefix = os.path.join(check_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator_model, critic=critic_model)

tot_epochs = 50

epoch = 0
tot_counter = 0
for epoch in range(tot_epochs):
    batch_counter = 0
    for train_batch, label_batch in zip(train_dataset_first, label_dataset_first):
        tot_counter = 5120*epoch+batch_counter
        print('Epoch {}, batch {}'.format(epoch,batch_counter))
        if batch_counter % 10 != 0:
            crit_loss, crit_gradients = training_steps.first_train_step_crit(train_batch, label_batch, 
                                                                            generator_model, critic_model, 
                                                                            p.first_crit_loss, p.first_critic_optimizer)
                            
        elif batch_counter % 10 == 0:
            (gen_loss, crit_loss, gen_gradients, crit_gradients) = training_steps.first_train_step(train_batch, label_batch,
                                                                            generator_model, critic_model, 
                                                                            p.first_gen_loss, p.first_crit_loss, 
                                                                            p.first_generator_optimizer, p.first_critic_optimizer)

        # for tensor in crit_gradients:
        #     parameters = 1
        #     for i in tensor.shape:
        #         parameters *= i
        #     print(parameters)

        img = generator_model.predict(test_slice)

        with train_summary_writer.as_default():
            tf.summary.scalar('critic loss', crit_loss, step=tot_counter)
            crit_tensor_counter=0
            for tensor in crit_gradients:
                tf.summary.histogram('critic gradients tensor {}'.format(crit_tensor_counter), tensor, step=tot_counter)
                crit_tensor_counter += 1

            crit_tensor_counter=0
            for tensor in critic_model.trainable_variables:
                tf.summary.histogram('critic tensor {}'.format(crit_tensor_counter), tensor, step=tot_counter)
                crit_tensor_counter += 1

            try:
                tf.summary.scalar('generator loss', gen_loss, step=tot_counter)
                gen_tensor_counter=0
                for tensor in gen_gradients:
                    tf.summary.histogram('generator gradients tensor {}'.format(gen_tensor_counter), tensor, step=tot_counter)
                    gen_tensor_counter += 1
            except:
                pass

            gen_tensor_counter=0
            for tensor in generator_model.trainable_variables:
                tf.summary.histogram('generator tensor {}'.format(gen_tensor_counter), tensor, step=tot_counter)
                gen_tensor_counter += 1

            tf.summary.image('sr slice', conv_uint(img), step=epoch)
            tf.summary.image('lr slice', conv_uint(test_slice), step=0)


            train_summary_writer.flush()
        batch_counter += 1
    epoch += 1


    # if batch_counter%10==0:
    #     checkpoint.save(file_prefix = checkpoint_prefix)
    #     np.savetxt(loss_dir+'gen_cost.txt', np.array(gen_cost_list))
    #     np.savetxt(loss_dir+'gen_cost_gan.txt', np.array(gen_cost_gan_list))
    #     np.savetxt(loss_dir+'gen_cost_inf.txt', np.array(gen_cost_inf_list))

    #     np.savetxt(loss_dir+'crit_cost.txt', np.array(crit_cost_list))
    #     np.savetxt(loss_dir+'crit_cost_sr.txt', np.array(crit_cost_sr_list))
    #     np.savetxt(loss_dir+'crit_cost_hr.txt', np.array(crit_cost_hr_list))
    #     np.savetxt(loss_dir+'crit_cost_grad.txt', np.array(crit_cost_grad_list))

    #     np.savetxt(grad_dir+'gen_grad.txt', np.array(grad_gen_list))
    #     np.savetxt(grad_dir+'gen_grad_gan.txt', np.array(grad_gen_gan_list))
    #     np.savetxt(grad_dir+'gen_grad_inf.txt', np.array(grad_gen_inf_list))

    #     np.savetxt(grad_dir+'crit_grad.txt', np.array(grad_crit_list))
    #     np.savetxt(grad_dir+'crit_grad_sr.txt', np.array(grad_sr_list))
    #     np.savetxt(grad_dir+'crit_grad_hr.txt', np.array(grad_hr_list))
    #     np.savetxt(grad_dir+'crit_grad_grad.txt', np.array(grad_grad_list))

