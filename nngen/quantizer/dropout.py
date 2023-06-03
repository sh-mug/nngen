from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as nps

def dropout(visitor, node):
    x, th, = node.args

    visitor.visit(x)
    visitor.visit(th)

    th_value = th.value
    node.scale_factor = x.scale_factor * (1 - th_value / (1 << 32))
