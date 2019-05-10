from . import HilbertSpace


class L2(HilbertSpace):
    """Space with default :math:`L^2` inner product.

    The Gram matrix is the identity.

    Arguments
    ---------
    shape : tuple of int
        Shape of array elements of the space.
    """

    def __init__(self, *shape):
        super().__init__(shape)

    @property
    def gram(self):
        return self.identity

    @property
    def gram_inv(self):
        return self.identity
