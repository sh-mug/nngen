from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

# import nngen as ng
import numpy as np
import nngen.basic_types as bt


class dropout(bt._ElementwiseOperator):
    """
    Applies dropout to the input by randomly setting a fraction of input units to 0
    at each update during training time, which helps prevent overfitting.

    Args:
        x: A Tensor.
        th: Used to set probability that each element is set to zero.
        seed: A Random Seed (optional).
    """
    
    input_chainable = True
    output_chainable = True

    def __init__(self, x, th, dtype=None, name=None, par=1, seed=0x12345678):
        self.seed = seed

        shape = None
        bt._ElementwiseOperator.__init__(self, x, th,
                                         dtype=dtype, shape=shape, name=name, par=par)

    def op(self, strm, *args, **kwargs):
        randval = strm.RandXorshift(reg_initval=self.seed)
        return strm.Mux(randval < args[1], strm.Int(0), args[0])

    def eval(self, memo, input_dict, **kwargs):
        kwargs['x_dtype'] = self.args[0].dtype
        kwargs['th_dtype'] = self.args[1].dtype
        kwargs['seed'] = self.seed
        return bt._ElementwiseOperator.eval(self, memo, input_dict, **kwargs)
