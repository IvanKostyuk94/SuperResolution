import numpy as np
import imageio as io

#draw_numbers = np.random.permutation(np.arange(512))
directory = '/home/ivkos/3dMultiPass/cubes_norm/'
k = 0
full_cube = np.load('/home/ivkos/Data/cube_normalized.npy')
for i in range(8):
    for j in range(8):
        for k in range(8):
            cube = full_cube[i*64:(i+1)*64, j*64:(j+1)*64, k*64:(k+1)*64]
            np.save(directory+'cube'+str(i*64+j*8+k), cube)

    