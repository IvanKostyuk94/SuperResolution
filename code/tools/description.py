# Short code returning the main properities of a given simulation cube

import numpy as np
import sys

name = str(sys.argv[1])
sr_cube = np.load(name)
print('The max value of the SR cube is {}'.format(np.amax(sr_cube)))
print('The min value of the SR cube is {}'.format(np.amin(sr_cube)))
print('The average value of the SR cube is {}'.format(np.average(sr_cube)))
print('The shape of the SR cube is {}'.format(sr_cube.shape))
