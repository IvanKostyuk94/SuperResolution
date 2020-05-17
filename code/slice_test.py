import numpy as np 
import tensorflow as tf
from models.test_model import *
import parameters as p
from matplotlib import pyplot as plt

# Define the generator and critic model for the first gen, crit
generator_input = layers.Input(shape=(None,None,1))
generator_model = tf.keras.Model(inputs = generator_input, outputs = generator(generator_input))

critic_input1 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
critic_input2 = layers.Input(shape=(p.x_dim_out, p.y_dim_out, 1))
critic_model = tf.keras.Model(inputs = (critic_input1, critic_input2), outputs = critic(critic_input1, critic_input2))

main_dir = '/u/ivkos/sr/'
run_dir = main_dir+'test_model_low_learning_rate/' 
ckpt_dir = run_dir+'checkpoints/'

latest = tf.train.latest_checkpoint(ckpt_dir)
checkpoint = tf.train.Checkpoint(generator=generator_model, critic = critic_model)
checkpoint.restore(latest)

def scale(data, epsilon=1e-9):
    return np.log(data+epsilon)/25
def unscale(data, epsilon=1e-9):
    return (np.exp(25*(data))-epsilon)

path_to_data = '/home/ivkos/cobra/sr/code/testing/testslice_lr.npy'
test_slice = np.load(path_to_data)
test_slice = scale(test_slice)
test_slice = test_slice.reshape(1, test_slice.shape[0], test_slice.shape[1],1)
prediction = generator_model.predict(test_slice)

#plt.figure()

#subplot(r,c) provide the no. of rows and columns
f, axarr = plt.subplots(2,1) 

# use the created array to output your multiple images. In this case I have stacked 4 images vertically
axarr[0].imshow(test_slice[0,:,:,0])
axarr[1].imshow(prediction[0,:,:,0])

plt.show()
