import logging

from . import Space

__all__ = ['L2']


class L2(Space):
    """Space with default L^2 inner product.

    The Gram matrix is the identity.

    Arguments
    ---------
    shape : tuple of int
        Shape of array elements of the space.
    """

    def __init__(self, *shape):
        super().__init__(shape, logging.getLogger(__name__))

    def gram(self, x):
        return x

    def gram_inv(self, x):
        return x
