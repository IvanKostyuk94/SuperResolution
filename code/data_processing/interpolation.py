import numpy as np 
from matplotlib import pyplot as plt

# Linear interpolation between images
def LinInterpol(stack):
    new_stack = np.zeros((stack.shape[0]*2, stack.shape[1], stack.shape[2]))
    for i in range(stack.shape[0]):
        if i == stack.shape[0]-1:
            new_stack[stack.shape[0]*2-2,:,:] = stack[i,:,:]
            new_stack[stack.shape[0]*2-1,:,:] = stack[i,:,:]
        else:
            try:
                new_stack[2*i, :, :] = stack[i, ...]
                new_stack[2*i+1, :, :] = (stack[i+1,:,:]+stack[i,:,:])/2.0
            except:
                print(i)
    return new_stack


directory = '/home/ivkos/3dMultiPass/'
for i in range(5120):
    cube = np.load(directory+'cubes_blurred2_norm/cube'+str(i)+'_blurred2.npy')
    interpol = LinInterpol(cube)
    print(interpol.shape)
    np.save(directory+'interpol_cubes_blurred2_norm/cube'+str(i)+'_blurred2.npy', interpol)
    # for j in range(64):

    #     ax = plt.subplot()
    #     im = ax.imshow(interpol[int(j),...])
    #     plt.savefig('interpol'+str(int(j))+'.png')
    #     plt.close()
    # exit()


