from . import Space
import numpy as np


class L2(Space):
    """Space with default :math:`L^2` inner product.

    The Gram matrix is the identity.

    Arguments
    ---------
    shape : tuple of int
        Shape of array elements of the space.
    """

    def __init__(self, coords):
        self.parameters_domain=start_l2(coords)
        super().__init__(np.size(coords), coords)


    def gram(self, x):
        return x

    def gram_inv(self, x):
        return x
    
class parameters_domain_l2:
    def __init__(self):
         return
     
def start_l2(coords):
    par_dom=parameters_domain_l2
    return par_dom
