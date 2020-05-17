import numpy as np 

draw_cubes = np.random.permutation(np.arange(512*10))
directory = '/home/ivkos/3dMultiPass/'
k = 0

for i in range(512):
    cube = np.load(directory+'cubes_norm/cube' + str(i) + '.npy')

    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), cube)
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.flip(cube,0))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.flip(cube,1))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.flip(cube,2))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.rot90(cube,1, axes = (0,1)))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.rot90(cube,2, axes = (0,1)))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.rot90(cube,3, axes = (0,1)))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.rot90(cube,1, axes = (1,2)))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.rot90(cube,2, axes = (1,2)))
    k += 1
    np.save(directory+'cubes_augmented_norm/cube' + str(draw_cubes[k]), np.rot90(cube,3, axes = (1,2)))
    k += 1