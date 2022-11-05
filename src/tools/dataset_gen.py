# Code used for generating training tiles out of gridded simulations
import numpy as np 

def create_dataset(simulation, random_array, output_dir, output_name):
    k = 0
    for i in range(8):
        for j in range(8):
            for l in range(8):
                cube = simulation[i*64:(i+1)*64, j*64:(j+1)*64, l*64:(l+1)*64]

                np.save(output_dir+output_name + str(random_array[k]), cube)
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]),np.flip(cube,0))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.flip(cube,1))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.flip(cube,2))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.rot90(cube,1, axes = (0,1)))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.rot90(cube,2, axes = (0,1)))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.rot90(cube,3, axes = (0,1)))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.rot90(cube,1, axes = (1,2)))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.rot90(cube,2, axes = (1,2)))
                k += 1
                np.save(output_dir+output_name+ str(random_array[k]), np.rot90(cube,3, axes = (1,2)))
                k += 1

lr_sim_in = np.load('/u/ivkos/sr/GriddedSimulationsTraining/256_grid512.npy')
hr_sim_in = np.load('/u/ivkos/sr/GriddedSimulationsTraining/512_grid512.npy')

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

def normalize_dataset(lr_sim, hr_sim, scaling, number):
    #lr_sim = scaling(lr_sim)
    #hr_sim = scaling(hr_sim)

    output_lr = '/u/ivkos/sr/TrainingTiles/lr_{}/'.format(number)
    output_hr = '/u/ivkos/sr/TrainingTiles/hr_{}/'.format(number)


    return(lr_sim, hr_sim, output_lr, output_hr)

lr_output_name = 'lr_cube'
hr_output_name = 'hr_cube'
# The random numbering of the individual training cubes
numbering = np.random.permutation(np.arange(8*8*8*10))

create_dataset(lr_sim_in, numbering, '/u/ivkos/sr/TrainingTiles/lr_no_scale/', lr_output_name)
create_dataset(hr_sim_in, numbering, '/u/ivkos/sr/TrainingTiles/hr_no_scale/', hr_output_name)
exit()

lr_sim, hr_sim, output_lr, output_hr = normalize_dataset(lr_sim_in, hr_sim_in, scale_2, 2)
create_dataset(lr_sim, numbering, output_lr, lr_output_name)
create_dataset(hr_sim, numbering, output_hr, hr_output_name)

lr_sim, hr_sim, output_lr, output_hr = normalize_dataset(lr_sim_in, hr_sim_in, scale_3, 3)
create_dataset(lr_sim, numbering, output_lr, lr_output_name)
create_dataset(hr_sim, numbering, output_hr, hr_output_name)

lr_sim, hr_sim, output_lr, output_hr = normalize_dataset(lr_sim_in, hr_sim_in, scale_4, 4)
create_dataset(lr_sim, numbering, output_lr, lr_output_name)
create_dataset(hr_sim, numbering, output_hr, hr_output_name)





