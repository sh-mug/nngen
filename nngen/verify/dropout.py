from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

randval = None

def gen_rand(seed):
    global randval

    if randval is None:
        randval = seed
        return randval
    else:
        randval = (randval ^ (randval << 13)) & ((1 << 32) - 1)
        randval = randval ^ ((randval >> 17) & ((1 << 32 - 17) - 1))
        randval = (randval ^ (randval << 5)) & ((1 << 32) - 1)
        return randval
            

def dropout(x, th, dtype=None, name=None, par=1,
            x_dtype=None, th_dtype=None, seed=np.array([0x12345678])):
    
    rand_array = np.array([gen_rand(seed) for _ in range(x.size)]) \
        .astype(np.uint32) \
        .reshape(x.shape)
    return np.where(rand_array < th, 0, x).astype(x.dtype)
